import os
import random
import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from models.abstract_model import AbstractGenerator
from PIL import Image
from torchvision import transforms
from util.dataclasses import DataclassType, DataShape, LabelDomain
from util.enums import DataSplit

from data.abstract_classes import AbstractDataset


class UTKFace(AbstractDataset):
    def __init__(self, root: str, image_size=200) -> None:
        super().__init__()
        self.root = root
        transformations = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        self.train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip()] + transformations
        )
        self.val_transform = transforms.Compose(transformations)

        self.images, self.labels = self._load_dataset()
        old_random_state = random.getstate()
        random.seed(0)
        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        self.images, self.labels = zip(*temp)
        random.setstate(old_random_state)

        num_images = len(self.labels)
        # Four splits to support training auxiliary classifiers, etc. 55-15-15-15
        self.len_train = int(np.floor(0.55 * num_images))
        self.len_val = int(np.floor(0.15 * num_images))
        self.len_test = int(np.floor(0.15 * num_images))
        self.len_holdout = int(np.ceil(0.15 * num_images))
        self.min_label = 0  # actual min is 1, but it's reasonable to include 0 years
        self.max_label = max(self.labels)

    def normalize_label(self, y: int) -> float:
        return (y - self.min_label) / (self.max_label - self.min_label)

    def denormalize_label(self, y: float) -> int:
        return int(y * (self.max_label - self.min_label) + self.min_label)

    def _load_dataset(self) -> Tuple[List[str], List[int]]:
        contents = os.listdir(self.root)
        label_pattern = re.compile(
            r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>[0-4])_(?P<timestamp>\d{17}).jpg"
        )
        images, labels = [], []
        for filename in contents:
            match = label_pattern.search(filename)
            if match:
                images.append(os.path.join(self.root, filename))
                labels.append(int(match.group("age")))
        return images, labels

    def random_targets(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape)

    def _len(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return self.len_train
        elif self.mode is DataSplit.VAL:
            return self.len_val
        elif self.mode is DataSplit.TEST:
            return self.len_test
        elif self.mode is DataSplit.HOLDOUT:
            return self.len_holdout

    def _get_idx_offset(self) -> int:
        if self.mode is DataSplit.TRAIN:
            return 0
        elif self.mode is DataSplit.VAL:
            return self.len_train
        elif self.mode is DataSplit.TEST:
            return self.len_train + self.len_val
        elif self.mode is DataSplit.HOLDOUT:
            return self.len_train + self.len_val + self.len_test

    def _getitem(self, index):
        index += self._get_idx_offset()
        filename, label = self.images[index], self.labels[index]
        image = Image.open(filename)
        if self.mode is DataSplit.TRAIN:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
        label = torch.tensor([label], dtype=torch.float32)
        image = self.normalize(image)
        label = self.normalize_label(label)
        return image, label

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        raise NotImplementedError()

    def data_shape(self) -> DataShape:
        x, y = self[0]
        return DataShape(y_dim=y.shape[0], n_channels=x.shape[0], x_size=x.shape[1])

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return None

    def has_performance_metrics(self) -> bool:
        return False

    def label_domain(self) -> Optional[LabelDomain]:
        return LabelDomain(0, 1)

