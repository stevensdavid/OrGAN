from collections import defaultdict
from typing import Tuple

import numpy as np
from torch import Tensor, tensor
from torch.utils.data import Dataset
from util.enums import DataSplit, VicinityType

from data.abstract_classes import AbstractDataset


class CcGANDatasetWrapper(AbstractDataset):
    def __init__(
        self, dataset: Dataset, type: VicinityType, sigma: float, hyperparam: float
    ) -> None:
        self.type = type
        self.dataset = dataset
        self.sigma = sigma
        self.labels = defaultdict(list)
        self.hyperparam = hyperparam
        for idx in range(len(self.dataset)):
            _, y = dataset[idx]
            self.labels[y].append(float(idx))
        self.unique_labels = np.asarray(self.labels.keys())

    def _getitem(self, _: int) -> Tuple[Tensor, Tensor]:
        sample_label = np.random.choice(self.unique_labels)
        noisy_label = sample_label + np.random.normal(scale=self.sigma)

        if self.type is VicinityType.HARD:
            lower_bound = sample_label - self.hyperparam
            upper_bound = sample_label + self.hyperparam
            candidate_labels = self.unique_labels[
                lower_bound <= self.unique_labels <= upper_bound
            ]
            candidate_idxs = [
                idx for label in candidate_labels for idx in self.labels[label]
            ]
            image_idx = np.random.choice(candidate_idxs)
            image, _ = self.dataset[image_idx]
            label = np.random.uniform(
                noisy_label - self.hyperparam, noisy_label + self.hyperparam
            )
            weight = 1
        elif self.type is VicinityType.SOFT:
            raise NotImplementedError("TODO")
        return image, {"labels": tensor(label), "loss_weights": tensor(weight)}

    def _len(self) -> int:
        return len(self.dataset)

    def set_mode(self, mode: DataSplit) -> None:
        self.dataset.set_mode(mode)
