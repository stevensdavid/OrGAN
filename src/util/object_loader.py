import yaml
from typing import Any
from inspect import signature
from pydoc import locate
from typing import Type


def load_yaml(filepath) -> dict:
    with open(filepath) as file:
        configuration = yaml.load(file, loader=yaml.FullLoader)
    return configuration


def build_from_yaml(filepath: str) -> Any:
    config = load_yaml(filepath)
    class_path: str = config["class_path"]
    class_object: Type = locate(class_path)
    try:
        instance = class_object(**config.kwargs)
    except TypeError:
        sig = signature(class_object)
        actual_kwargs = config.kwargs.keys()
        expected_kwargs = sig.parameters.keys()
        missing_kwargs = expected_kwargs - actual_kwargs
        unexpected_kwargs = actual_kwargs - expected_kwargs
        raise ValueError(
            f"'kwargs' mapping in {filepath} contains errors.\n"
            + f"Missing kwargs: {', '.join(missing_kwargs)}\n"
            + f"Unexpected kwargs: {', '.join(unexpected_kwargs)}"
        )
    return instance
