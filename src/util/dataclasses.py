from __future__ import annotations

from operator import add, sub
from dataclasses import astuple, dataclass
from typing import Dict
from typing_extensions import Protocol
from util.enums import FrequencyMetric
from util.object_loader import load_yaml


class DataclassType(Protocol):
    # Method from https://stackoverflow.com/a/55240861
    # checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict


class DataclassExtensions:
    def __add__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(add, astuple(self), astuple(other))))

    def __add__(self: DataclassType, term: float):
        return type(self)(*tuple(map(lambda x: x + term, astuple(self))))

    def __sub__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(sub, astuple(self), astuple(other))))

    def __truediv__(self: DataclassType, denominator: float):
        return type(self)(*tuple(map(lambda x: x / denominator, astuple(self))))


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
