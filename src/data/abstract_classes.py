from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torch import tensor, Size

class AbstractDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def random_targets(self, shape: Size) -> tensor:
        ...
