from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from operator import add, mul, sub
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from typing_extensions import Protocol

from util.enums import FrequencyMetric, MultiGPUType
from util.object_loader import load_yaml


@dataclass
class Metric:
    mean: float
    stddev: float

    @staticmethod
    def from_list(l: List):
        return Metric(mean=np.mean(l), stddev=np.std(l))


class DataclassType(Protocol):
    # Method from https://stackoverflow.com/a/55240861
    # checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict


class DataclassExtensions:
    def to_tuple(self):
        return tuple(getattr(self, field.name) for field in fields(self))

    def __add__(self: DataclassType, term: Union[DataclassType, float]):
        if isinstance(term, DataclassExtensions):
            return type(self)(*tuple(map(add, self.to_tuple(), term.to_tuple())))
        else:
            return type(self)(*tuple(map(lambda x: x + term, self.to_tuple())))

    def __mul__(self: DataclassType, term: Union[DataclassType, float]):
        if isinstance(term, DataclassExtensions):
            return type(self)(*tuple(map(mul, self.to_tuple(), term.to_tuple())))
        else:
            return type(self)(*tuple(map(lambda x: x * term, self.to_tuple())))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(sub, self.to_tuple(), other.to_tuple())))

    def __truediv__(self: DataclassType, denominator: float):
        return type(self)(*tuple(map(lambda x: x / denominator, self.to_tuple())))

    def to_plain_datatypes(self) -> DataclassType:
        return type(self)(*tuple(map(lambda x: x.item(), self.to_tuple())))

    def to_tensor(self) -> torch.Tensor:
        return torch.stack(self.to_tuple())

    def from_tensor(self, t: torch.Tensor) -> DataclassType:
        return type(self)(*t)

    def map(self, op: Callable) -> DataclassType:
        return type(self)(*tuple(map(op, self.to_tuple())))

    def extend(self, other: DataclassType) -> None:
        for field in fields(self):
            getattr(self, field.name).extend(getattr(other, field.name))


@dataclass
class DataShape:
    y_dim: int
    n_channels: int
    x_size: int
    embedding_dim: int = None


@dataclass
class TrainingConfig:
    checkpoint_frequency: int
    checkpoint_frequency_metric: FrequencyMetric
    log_frequency: int
    log_frequency_metric: FrequencyMetric
    multi_gpu_type: MultiGPUType
    skip_validation: bool = False
    sample_frequency: int = None

    @staticmethod
    def from_yaml(filepath: str) -> TrainingConfig:
        cfg = load_yaml(filepath)
        cfg["checkpoint_frequency_metric"] = (
            FrequencyMetric.ITERATIONS
            if cfg["checkpoint_frequency_metric"] == "iterations"
            else FrequencyMetric.EPOCHS
        )
        cfg["log_frequency_metric"] = (
            FrequencyMetric.ITERATIONS
            if cfg["log_frequency_metric"] == "iterations"
            else FrequencyMetric.EPOCHS
        )
        cfg["multi_gpu_type"] = (
            MultiGPUType.DDP
            if cfg["multi_gpu_type"] == "ddp"
            else MultiGPUType.DATA_PARALLEL
        )
        return TrainingConfig(**cfg)


@dataclass
class GeneratedExamples:
    image: torch.tensor
    label: str


@dataclass
class LabelDomain:
    min: float
    max: float


if __name__ == "__main__":

    @dataclass
    class A(DataclassExtensions):
        x: torch.Tensor
        y: torch.Tensor
        z: torch.Tensor

    a = A(torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]))
    t = a.to_tensor()
    b = a.from_tensor(t)
    assert torch.equal(b.x, a.x)
    assert torch.equal(b.y, a.y)
    assert torch.equal(b.z, a.z)
