from typing import Tuple
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from torch import Tensor

from util.enums import VicinityType


class CcGANDatasetWrapper(Dataset):
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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample_label = np.random.choice(self.unique_labels)
        noisy_label = sample_label + np.random.normal(scale=self.sigma)
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.dataset)
