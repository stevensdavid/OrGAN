from inspect import signature
from pydoc import locate
from typing import Any, Type

import yaml


def qualified_type_name(t: Any) -> str:
    t = type(t)
    return f"{t.__module__}.{t.__name__}"


def load_yaml(filepath) -> dict:
    with open(filepath) as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    return configuration


def build_from_yaml(filepath: str, **kwargs) -> Any:
    config = load_yaml(filepath)
    class_object: Type = locate(config["class"])
    build_kwargs = {**kwargs, **config["kwargs"]}
    try:
        instance = class_object(**build_kwargs)
    except TypeError as e:
        error_msg = e.args[0]
        if not error_msg.startswith("__init__() got an unexpected keyword argument"):
            raise e
        sig = signature(class_object.__init__)
        actual_kwargs = config["kwargs"].keys()
        expected_kwargs = sig.parameters.keys()
        missing_kwargs = expected_kwargs - actual_kwargs - {"self"}
        unexpected_kwargs = actual_kwargs - expected_kwargs
        raise ValueError(
            f"'kwargs' mapping in {filepath} contains errors.\n"
            + f"Missing kwargs: {', '.join(missing_kwargs)}\n"
            + f"Unexpected kwargs: {', '.join(unexpected_kwargs)}"
        ) from e
    return instance
