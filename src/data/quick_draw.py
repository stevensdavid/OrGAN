import logging
import os
from typing import List, Tuple

import numpy as np
import requests
import torch
from torch import nn
from util.dataclasses import DataShape, LabelDomain
from util.enums import DataSplit

from data.abstract_classes import AbstractDataset


class BinaryQuickDraw(AbstractDataset):
    def __init__(
        self, root: str, classes: List[str] = ["cat", "dog"], normalize: bool = True
    ) -> None:
        super().__init__()
        self.log = logging.getLogger("QuickDraw")
        self.root = root
        self.classes = classes
        self.normalize_images = normalize
        self.n_classes = len(classes)
        self.max_label = self.n_classes - 1
        if not os.path.exists(root):
            self._download()
        self.images = self._load_images()
        self.labels = self._load_labels()
        # Deterministic shuffle
        old_random_state = np.random.get_state()
        np.random.seed(0)
        order = np.random.permutation(len(self.labels))
        self.images = self.images[order]
        self.labels = self.labels[order]
        np.random.set_state(old_random_state)
        # Train-val-test split is 70-15-15
        n_images = len(self.labels)
        self.len_train = int(np.floor(0.7 * n_images))
        self.len_val = int(np.floor(0.15 * n_images))
        self.len_test = int(np.ceil(0.15 * n_images))
        self.pad = nn.ZeroPad2d(2)

    def _load_images(self) -> np.ndarray:
        return np.load(self._get_image_path())

    def _load_labels(self) -> np.ndarray:
        return np.load(self._get_label_path())

    def _get_image_path(self) -> str:
        return os.path.join(self.root, "images.npy")

    def _get_label_path(self) -> str:
        return os.path.join(self.root, "labels.npy")

    def _download(self):
        self.log.info("Downloading QuickDraw")
        base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
        all_images = None
        all_labels = None
        os.makedirs(self.root, exist_ok=True)
        for label, class_name in enumerate(self.classes):
            self.log.info(
                f"Downloading class: {class_name} ({label + 1}/{self.n_classes})"
            )
            response = requests.get(
                f"{base_url}/{class_name}.npy", allow_redirects=True
            )
            temp_path = os.path.join(self.root, f"raw_images_{class_name}.npy")
            with open(temp_path, "wb") as f:
                f.write(response.content)
            images = np.load(temp_path)
            images = images.reshape((-1, 28, 28))
            labels = np.ones(images.shape[0]) * label
            if all_images is None:
                all_images = images
                all_labels = labels
            else:
                all_images = np.concatenate((all_images, images))
                all_labels = np.concatenate((all_labels, labels))
        np.save(self._get_image_path(), all_images)
        np.save(self._get_label_path(), all_labels)

    def label_domain(self) -> LabelDomain:
        return LabelDomain(0, self.max_label)

    def _get_idx_offset(self):
        if self.mode is DataSplit.TRAIN:
            return 0
        elif self.mode is DataSplit.VAL:
            return self.len_train
        elif self.mode is DataSplit.TEST:
            return self.len_train + self.len_val

    def _getitem(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = self._get_idx_offset()
        index += offset
        x = self.images[index]
        x = x / 255
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.unsqueeze(x, 0)  # Add channel dimension
        x = self.pad(x)
        if self.normalize_images:
            x = self.normalize(x)
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float32)
        y = torch.unsqueeze(y, 0)
        return x, y

    def _len(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return self.len_train
        elif self.mode is DataSplit.TEST:
            return self.len_test
        elif self.mode is DataSplit.VAL:
            return self.len_val

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape) * self.max_label

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=y.shape[0], n_channels=x.shape[0], x_size=x.shape[1])

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> None:
        return None

    def has_performance_metrics(self) -> bool:
        return False


if __name__ == "__main__":
    dataset = BinaryQuickDraw("data/quickdraw")
    import matplotlib.pyplot as plt

    dataset.set_mode(DataSplit.TRAIN)

    n_images = 10
    for idx in range(n_images):
        ax = plt.subplot(1, n_images, idx + 1)
        x, y = dataset[idx]
        x = x.numpy()
        # remove dimension axis
        x = np.squeeze(x)
        y = y.item()
        ax.imshow(x)
        ax.axis("off")
        ax.title.set_text(f"Class {int(y)}")
    plt.tight_layout()
    plt.show()
