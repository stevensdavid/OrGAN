from collections import defaultdict
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Size, Tensor, tensor
from torch.utils.data import Dataset
from models.abstract_model import AbstractGenerator
from util.dataclasses import DataShape, DataclassType, LabelDomain
from util.enums import DataSplit, VicinityType

from data.abstract_classes import AbstractDataset


class CcGANDatasetWrapper(AbstractDataset):
    def __init__(
        self,
        dataset: Dataset,
        type: VicinityType,
        sigma: float,
        n_neighbours: int,
        clip=True,
    ) -> None:
        self.type = type
        self.dataset = dataset
        self.sigma = sigma
        self.labels = defaultdict(list)

        self.clip = clip
        for idx in range(len(self.dataset)):
            _, y = dataset[idx]
            self.labels[y.item()].append(idx)
        self.unique_labels = np.asarray(sorted(self.labels.keys()))
        self.min_label = np.min(self.unique_labels)
        self.max_label = np.max(self.unique_labels)
        # Set according to rule of thumb from CCGAN paper appendix S.9
        kappa_base = np.max(self.unique_labels[1:] - self.unique_labels[:-1])
        kappa = n_neighbours * kappa_base
        if type is VicinityType.HARD:
            self.hyperparam = kappa
        elif type is VicinityType.SOFT:
            self.hyperparam = 1 / (kappa ** 2)
        self.eps = 10e-3
        self.soft_offset = np.sqrt(-(np.log(self.eps) / self.hyperparam))

    def svdl_distance(self, label, noisy_label) -> float:
        return np.exp(-self.hyperparam * (label - noisy_label) ** 2)

    def _getitem(self, _: int) -> Tuple[Tensor, Tensor]:
        sample_label = np.random.choice(self.unique_labels)
        noisy_label = sample_label + np.random.normal(scale=self.sigma)
        if self.clip:
            noisy_label = np.clip(noisy_label, self.min_label, self.max_label)

        if self.type is VicinityType.HARD:
            candidate_labels = self.unique_labels[
                np.abs(self.unique_labels - noisy_label) <= self.hyperparam
            ]
        elif self.type is VicinityType.SOFT:
            candidate_labels = self.unique_labels[
                self.svdl_distance(self.unique_labels, noisy_label) > self.eps
            ]
        candidate_idxs = [
            idx for label in candidate_labels for idx in self.labels[label]
        ]
        image_idx = np.random.choice(candidate_idxs)
        image, label = self.dataset[image_idx]
        label = label.item()
        if self.type is VicinityType.HARD:
            weight = 1
            target_label = np.random.uniform(
                noisy_label - self.hyperparam, noisy_label + self.hyperparam
            )
            target_weight = 1
        elif self.type is VicinityType.SOFT:
            weight = self.svdl_distance(label, noisy_label)
            target_label = np.random.uniform(
                noisy_label - self.soft_offset, noisy_label + self.soft_offset
            )
            target_weight = self.svdl_distance(noisy_label, target_label)
        if self.clip:
            target_label = np.clip(target_label, self.min_label, self.max_label)
        return (
            image,
            {
                "labels": tensor([noisy_label], dtype=torch.float32),
                "label_weights": tensor(weight, dtype=torch.float32),
                "target_labels": tensor([target_label], dtype=torch.float32),
                "target_weights": tensor(target_weight, dtype=torch.float32),
            },
        )

    def random_targets(self, shape: Size) -> tensor:
        return self.dataset.random_targets(shape)

    def data_shape(self) -> DataShape:
        return self.dataset.data_shape()

    def _len(self) -> int:
        return len(self.dataset)

    def set_mode(self, mode: DataSplit) -> None:
        super().set_mode(mode)
        self.dataset.set_mode(mode)

    def performance(self, real_images, real_labels, fake_images, fake_labels) -> dict:
        return self.dataset.performance(
            real_images, real_labels, fake_images, fake_labels
        )

    def label_domain(self) -> Optional[LabelDomain]:
        return self.dataset.label_domain()

    def test_model(
        self,
        generator: AbstractGenerator,
        batch_size: int,
        n_workers: int,
        device: torch.device,
        label_transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> DataclassType:
        return self.dataset.test_model(
            generator, batch_size, n_workers, device, label_transform
        )
