from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import skimage.color
import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import CenterCrop
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
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.mode = DataSplit.TRAIN if train else DataSplit.TEST
        # FashionMNIST is split into train and test.
        # Create validation split if we are training
        total_samples = super().__len__()
        if train:
            self.len_train = int(total_samples * 0.8)
            self.len_val = total_samples - self.len_train
        # Generate random labels *once*
        np.random.seed(0)
        ys = np.linspace(0, 1, num=10, endpoint=False)
        ys = np.repeat(ys, total_samples // len(ys))
        np.random.shuffle(ys)
        self.ys = ys
        self.hues = (ys + np.random.normal(size=ys.shape, scale=1 / 100)) % 1
        self.pad = nn.ZeroPad2d(2)

    def _getitem(self, index):
        return self[index]

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        if self.mode is DataSplit.VAL:
            # We use first `self.len_train` samples for training set, so offset index
            index += self.len_train
        x, _ = super().__getitem__(index)
        x = np.array(x, dtype=float)
        y = self.ys[index]
        hue = self.hues[index]
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
        return torch.rand(shape)

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=1, n_channels=x.shape[0], x_size=x.shape[1])

    @staticmethod
    def shift_hue(image: np.ndarray, factor: float) -> np.ndarray:
        """Shift the hue of an image

        Args:
            pil_image (Image): Grayscale image
            factor (float): Between 0 and 1, the HSV hue value to shift by

        Returns:
            np.ndarray: Hue-shifted image
        """
        x = skimage.color.gray2rgb(image)
        x = skimage.color.rgb2hsv(x)
        # Shift hue in HSV
        x[:, :, 0] += factor.item() if isinstance(factor, torch.Tensor) else factor
        x[:, :, 0] %= 1
        # Saturate grayscale
        x[:, :, 1] = 1
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)  # Move channels to front
        x /= 255
        return x

    @staticmethod
    def ground_truth(x: np.ndarray, y: float) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = np.moveaxis(x, 0, -1)
        x_prime = skimage.color.rgb2gray(x)
        x_prime = HSVFashionMNIST.shift_hue(x, y)
        return x_prime

    @staticmethod
    def ground_truths(xs: List[np.ndarray], ys: List[float]) -> List[np.ndarray]:
        return [HSVFashionMNIST.ground_truth(x, y) for x, y in zip(xs, ys)]


if __name__ == "__main__":
    dataset = HSVFashionMNIST("FashionMNIST/", download=True)
    import matplotlib.pyplot as plt

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
