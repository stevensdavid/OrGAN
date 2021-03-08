from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from torch import tensor, Size
from util.dataclasses import DataShape
from util.enums import DataSplit


class AbstractDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def random_targets(self, shape: Size) -> tensor:
        ...

    def set_mode(
        self,
        mode: DataSplit,
    ) -> None:
        self.mode = mode

    def __len__(self) -> int:
        if self.mode is None:
            raise ValueError("Please call 'set_mode' before using data set")
        return self._len()

    def __getitem__(self, index: int):
        if self.mode is None:
            raise ValueError("Please call 'set_mode' before using data set")
        return self._getitem(index)

    @abstractmethod
    def _len(self) -> int:
        ...

    @abstractmethod
    def _getitem(self, index):
        ...

    @abstractmethod
    def data_shape(self) -> DataShape:
        ...
