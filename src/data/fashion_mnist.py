from typing import Any, Callable, Optional, Tuple

import numpy as np
import skimage.color
import torch
from torchvision.datasets import FashionMNIST
from util.dataclasses import DataShape

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, _ = super().__getitem__(index)
        x = np.array(x, dtype=float)
        y = np.random.rand()
        x = self.shift_hue(x, y)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )

    def _getitem(self, _):
        raise NotImplementedError("Call __getitem__ directly")

    def _len(self):
        raise NotImplementedError("Call __len__ directly")

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
        x[:, :, 0] += factor
        x[:, :, 0] %= 1
        # Saturate grayscale
        x[:, :, 1] = 1
        x = skimage.color.hsv2rgb(x)
        x = np.moveaxis(x, -1, 0)  # Move channels to front
        return x

    @staticmethod
    def ground_truth(x: np.ndarray, y: float) -> np.ndarray:
        x_prime = skimage.color.rgb2gray(x)
        x_prime = HSVFashionMNIST.shift_hue(x, y)
        return x_prime


if __name__ == "__main__":
    dataset = HSVFashionMNIST("FashionMNIST/", download=True)
    import matplotlib.pyplot as plt

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        x, y = dataset[0]
        x = np.moveaxis(x, 0, -1)
        ax.imshow(x)
        ax.axis("off")
        ax.title.set_text(f"H={y:.3f}")
    plt.show()
