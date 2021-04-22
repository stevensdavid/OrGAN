from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import skimage.color
import torch
import torch.linalg
from torch import nn
from torchvision.datasets import FashionMNIST
from util.dataclasses import DataclassExtensions, DataShape, GeneratedExamples
from util.enums import DataSplit, ReductionType

from data.abstract_classes import AbstractDataset


@dataclass
class HSVFashionMNISTPerformance(DataclassExtensions):
    hsv_mae_h: torch.Tensor
    hsv_mae_s: torch.Tensor
    hsv_mae_v: torch.Tensor
    hsv_l1: torch.Tensor
    hsv_l2: torch.Tensor
    rgb_l1: torch.Tensor
    rgb_l2: torch.Tensor


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
        # Scale to [-1,1]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        x = self.pad(x)  # Zero-pads 28x28 to 32x32
        x = self.normalize(x)
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

    def ground_truth(self, x: np.ndarray, y: float) -> np.ndarray:
        if x.shape[0] != 3:
            raise AssertionError(f"Incorrect shape in ground_truth: {x.shape}")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Scale to [0,1]
        x = self.denormalize(x)
        x = np.moveaxis(x, 0, -1)
        x = skimage.color.rgb2hsv(x)
        x[:, :, 0] = y
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)
        # Scale back to [-1, 1]
        x = self.normalize(x)
        return x

    def ground_truths(self, xs: List[np.ndarray], ys: List[float]) -> np.ndarray:
        return np.asarray([self.ground_truth(x, y) for x, y in zip(xs, ys)])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Scale from [0,1] to [-1,1]

        Args:
            x (np.ndarray): Image with values in range [0,1]

        Returns:
            np.ndarray: Image with values in range [-1,1]
        """
        return 2 * x - 1

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Scale from [-1,1] to [0,1]

        Args:
            x (np.ndarray): Image with values in range [-1,1]

        Returns:
            np.ndarray: Image with values in range [0,1]
        """
        return (x + 1) / 2

    def rgb_tensor_to_hsv(self, t: torch.Tensor) -> torch.Tensor:
        x = t.cpu().numpy()
        x = self.denormalize(x)
        x = np.moveaxis(x, 1, -1)
        x = skimage.color.rgb2hsv(x)
        x = np.moveaxis(x, -1, 0)
        return torch.tensor(x, device=t.device)

    def performance(
        self,
        real_images,
        real_labels,
        fake_images,
        fake_labels,
        reduction: ReductionType,
    ) -> HSVFashionMNISTPerformance:
        ground_truths = self.ground_truths(real_images, fake_labels)
        ground_truths = torch.tensor(ground_truths, device=fake_images.device)
        rgb_l1 = torch.mean(torch.abs(ground_truths - fake_images), dim=0)
        rgb_l2 = torch.linalg.norm(ground_truths - fake_images, dim=0)
        hsv_truths = self.rgb_tensor_to_hsv(ground_truths)
        hsv_fakes = self.rgb_tensor_to_hsv(fake_images)
        h_error = torch.mean(
            torch.abs(hsv_truths[:, 0, :, :] - hsv_fakes[:, 0, :, :]), dim=0
        )
        s_error = torch.mean(
            torch.abs(hsv_truths[:, 1, :, :] - hsv_fakes[:, 1, :, :]), dim=0
        )
        v_error = torch.mean(
            torch.abs(hsv_truths[:, 2, :, :] - hsv_fakes[:, 2, :, :]), dim=0
        )
        hsv_l1 = torch.mean(torch.abs(hsv_truths - hsv_fakes), dim=0)
        hsv_l2 = torch.mean(torch.linalg.norm(hsv_truths - hsv_fakes, dim=0), dim=0)

        return_values = HSVFashionMNISTPerformance(
            h_error, s_error, v_error, hsv_l1, hsv_l2, rgb_l1, rgb_l2
        )
        if reduction is ReductionType.MEAN:
            reduce = lambda x: torch.mean(x)
        elif reduction is ReductionType.SUM:
            reduce = lambda x: torch.sum(x)
        elif reduction is ReductionType.NONE:
            reduce = lambda x: x
        return_values = return_values.map(reduce)
        return return_values

    def stitch_examples(self, real_images, real_labels, fake_images, fake_labels):
        def stitch_image(real, fake, target_label):
            truth = self.ground_truth(real, target_label)
            merged = np.concatenate((real, fake, truth), axis=2)
            return np.moveaxis(merged, 0, -1)

        return [
            GeneratedExamples(stitch_image(real, fake, target), f"H={target}")
            for real, fake, target in zip(real_images, fake_images, fake_labels)
        ]


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
