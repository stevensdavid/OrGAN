from operator import add, sub
from dataclasses import astuple
from typing import Dict
from typing_extensions import Protocol


class DataclassType(Protocol):
    # Method from https://stackoverflow.com/a/55240861
    # checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict


class DataclassArithmetic:
    def __add__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(add, astuple(self), astuple(other))))

    def __sub__(self: DataclassType, other: DataclassType):
        return type(self)(*tuple(map(sub, astuple(self), astuple(other))))

    def __truediv__(self: DataclassType, denominator: float):
        return type(self)(*tuple(map(lambda x: x/denominator, astuple(self))))
