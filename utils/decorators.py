# utils/decorators.py
import functools
from pathlib import Path


def validate_dataset_path(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        path = Path(self.root)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path {path} does not exist")
        if not any(path.iterdir()):
            raise ValueError(f"Dataset path {path} is empty")
        return func(*args, **kwargs)

    return wrapper
