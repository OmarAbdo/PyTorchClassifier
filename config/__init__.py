# config/__init__.py
import os
import yaml
from typing import Dict, Any


class AppConfig:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls) -> None:
        with open("config/settings.yaml", "r") as f:
            base_config = yaml.safe_load(f)

        # Allow environment variable overrides
        cls.config: Dict[str, Any] = {
            **base_config,
            **os.environ,  # Environment variables take precedence
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


# Usage: config = AppConfig()
