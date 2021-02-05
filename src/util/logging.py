def add_key_prefix(dictionary: dict, prefix: str, separator: str = "/") -> dict:
    return {f"{prefix}{separator}{k}": v for k, v in dictionary.items()}
