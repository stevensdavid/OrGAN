from torchvision.datasets import FashionMNIST
from typing import Optional, Callable, Tuple, Any

import numpy as np
import skimage.color
from PIL import Image


def shift_hue(image: Image, factor: float) -> np.ndarray:
    """Shift the hue of an image

    Args:
        pil_image (Image): Grayscale image
        factor (float): Between 0 and 1, the HSV hue value to shift by

    Returns:
        np.ndarray: Hue-shifted image
    """
    x = np.array(image)
    x = skimage.color.gray2rgb(x)
    x = skimage.color.rgb2hsv(x)
    # Shift hue in HSV
    x[:, :, 0] += factor
    x[:, :, 0] %= 1
    # Saturate grayscale
    x[:, :, 1] = 1
    x = skimage.color.hsv2rgb(x)
    x = np.moveaxis(x, -1, 0)  # Move channels to front
    return x


class HSVFashionMNIST(FashionMNIST):
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
        y = np.random.rand()
        x = shift_hue(x, y)
        return x, y


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
