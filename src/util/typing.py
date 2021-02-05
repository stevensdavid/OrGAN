from typing import Dict
from typing_extensions import Protocol


class DataclassType(Protocol):
    # Method from https://stackoverflow.com/a/55240861
    # checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: Dict
