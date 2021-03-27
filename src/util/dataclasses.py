from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from operator import add, sub
from typing import Dict, Union

from typing_extensions import Protocol

from util.enums import FrequencyMetric
from util.object_loader import load_yaml


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

    def __sub__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(sub, self.to_tuple(), other.to_tuple())))

    def __truediv__(self: DataclassType, denominator: float):
        return type(self)(*tuple(map(lambda x: x / denominator, self.to_tuple())))

    def to_plain_datatypes(self) -> DataclassType:
        return type(self)(*tuple(map(lambda x: x.item(), self.to_tuple())))


@dataclass
class DataShape:
    y_dim: int
    n_channels: int
    x_size: int


@dataclass
class TrainingConfig:
    checkpoint_frequency: int
    checkpoint_frequency_metric: FrequencyMetric
    log_frequency: int
    log_frequency_metric: FrequencyMetric

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
        return TrainingConfig(**cfg)