from typing import Dict, Type
import torchvision.models as tv_models
from config import AppConfig
from .base_model import BaseAnimalModel


class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str) -> callable:
        def decorator(model_class: Type[BaseAnimalModel]):
            if name in cls._registry:
                raise ValueError(f"Model {name} already registered")
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, name: str, config: Dict) -> BaseAnimalModel:
        model_class = cls._registry.get(name)
        if not model_class:
            raise ValueError(f"Model {name} not in registry")
        return model_class(config)


class AnimalModelFactory:
    def __init__(self):
        self.config = AppConfig()

    def get_model(self, version: str = None) -> BaseAnimalModel:
        version = version or self.config.get("models.current_version")
        model_config = self.config.get(f"models.versions.{version}", {})
        base_config = self.config.get("models.base", {})

        return ModelRegistry.create_model(
            name=base_config["architecture"], config={**base_config, **model_config}
        )
