from typing import Callable, List, Optional, Tuple

import numpy as np
import skimage.color
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from util.dataclasses import DataShape
from util.enums import DataSplit

from data.abstract_classes import AbstractDataset


class HSVFashionMNIST(FashionMNIST, AbstractDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        simplified: bool = False,
        n_clusters: Optional[int] = None,
        noisy_labels: bool = True,
        fixed_labels: bool = True,
        min_hue: float = 0.0,
        max_hue: float = 1.0,
        cyclical: bool = True,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.mode = DataSplit.TRAIN if train else DataSplit.TEST
        self.simplified = simplified
        self.fixed_labels = fixed_labels
        self.min_hue = min_hue
        self.max_hue = max_hue
        self.cyclical = cyclical
        # FashionMNIST is split into train and test.
        # Create validation split if we are training
        total_samples = super().__len__()
        if train:
            self.len_train = int(total_samples * 0.8)
            self.len_val = total_samples - self.len_train
        # Generate random labels *once*
        np.random.seed(0)
        self.n_clusters = n_clusters
        self.noisy_labels = noisy_labels
        if n_clusters is None:
            self.hues = np.random.uniform(self.min_hue, self.max_hue, total_samples)
            self.ys = self.hues
            if noisy_labels:
                self.ys += np.random.normal(size=self.ys.shape, scale=1e-2)
                if cyclical:
                    self.ys %= 1
        else:
            ys = np.linspace(self.min_hue, self.max_hue, num=n_clusters, endpoint=False)
            ys = np.repeat(ys, total_samples // len(ys))
            np.random.shuffle(ys)
            self.ys = ys
            self.hues = ys + np.random.normal(
                size=ys.shape, scale=(self.max_hue - self.min_hue) / (n_clusters * 10)
            )
            if self.cyclical:
                self.hues %= 1
        self.pad = nn.ZeroPad2d(2)

    def _getitem(self, index):
        return self[index]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        if self.mode is DataSplit.VAL:
            # We use first `self.len_train` samples for training set, so offset index
            index += self.len_train
        x, _ = super().__getitem__(index)
        x = np.array(x, dtype=float)
        x /= 255  # Normalize for hue shift
        if self.fixed_labels:
            y = self.ys[index]
            hue = self.hues[index]
        else:
            if self.n_clusters is not None:
                idx = np.random.randint(0, len(self.ys))
                y = self.ys[idx]
                hue = self.hues[idx]
            else:
                y = np.random.uniform(self.min_hue, self.max_hue)
                hue = y
                if self.noisy_labels:
                    raise ValueError("Noisy labels not supported without fixed labels.")
        x = self.shift_hue(x, hue)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        x = self.pad(x)  # Zero-pads 28x28 to 32x32
        return x, y

    def _len(self) -> int:
        return len(self)

    def __len__(self):
        if self.mode is DataSplit.TRAIN:
            return self.len_train
        elif self.mode is DataSplit.VAL:
            return self.len_val
        elif self.mode is DataSplit.TEST:
            return len(self)

    def random_targets(self, shape: torch.Size) -> torch.tensor:
        if self.simplified:
            # Only include labels that are part of the dataset
            return torch.tensor(
                np.random.choice(self.ys, size=shape), dtype=torch.float32
            )
        return (self.max_hue - self.min_hue) * torch.rand(shape) + self.min_hue

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=1, n_channels=x.shape[0], x_size=x.shape[1])

    @staticmethod
    def shift_hue(image: np.ndarray, factor: float) -> np.ndarray:
        """Shift the hue of an image

        Args:
            image (np.ndarray): Grayscale image, between 0 and 1
            factor (float): Between 0 and 1, the HSV hue value to shift by

        Returns:
            np.ndarray: Hue-shifted image
        """
        if len(image.shape) != 2:
            raise AssertionError(f"Incorrect shape in shift_hue: {image.shape}")
        x = skimage.color.gray2rgb(image)
        x = skimage.color.rgb2hsv(x)
        # Shift hue in HSV
        x[:, :, 0] = factor.item() if isinstance(factor, torch.Tensor) else factor
        # x[:, :, 0] %= 1 # Should be redundant as hues are [0,1)
        # Saturate grayscale
        x[:, :, 1] = 1
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)  # Move channels to front
        return x

    @staticmethod
    def ground_truth(x: np.ndarray, y: float) -> np.ndarray:
        if x.shape[0] != 3:
            raise AssertionError(f"Incorrect shape in ground_truth: {x.shape}")
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = np.moveaxis(x, 0, -1)
        x = skimage.color.rgb2hsv(x)
        x[:, :, 0] = y
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)
        return x

    @staticmethod
    def ground_truths(xs: List[np.ndarray], ys: List[float]) -> List[np.ndarray]:
        return [HSVFashionMNIST.ground_truth(x, y) for x, y in zip(xs, ys)]


if __name__ == "__main__":
    dataset = HSVFashionMNIST("FashionMNIST/", download=True)
    x = dataset.random_targets((30, 1))
    import matplotlib.pyplot as plt

    x, y = dataset[0]
    z = dataset.ground_truth(x, y, y)
    z2 = dataset.ground_truth(x, y, 0.5)
    plt.figure(0)
    combined = np.concatenate((x, z, z2), axis=1)
    combined = np.moveaxis(combined, 0, -1)
    plt.imshow(combined)

    plt.figure(1)
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        x, y = dataset[i]
        x, y = x.numpy(), y.numpy()
        x = np.moveaxis(x, 0, -1)
        ax.imshow(x)
        ax.axis("off")
        ax.title.set_text(f"H={y.item():.3f}")
    plt.tight_layout()
    plt.figure(2)
    plt.hist(dataset.ys, bins=100)
    plt.show()
