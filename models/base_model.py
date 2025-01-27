from abc import ABC, abstractmethod
from typing import Dict, List
import torch.nn as nn
import torch


class ModelConfigurationError(Exception):
    pass


class BaseAnimalModel(nn.Module, ABC):
    def __init__(self, config: Dict):
        super().__init__()
        self._validate_config(config)

    @abstractmethod
    def freeze_layers(self, layer_names: List[str]) -> None:
        pass

    @abstractmethod
    def get_classifier(self) -> nn.Module:
        pass

    @abstractmethod
    def get_optimizer(self, params) -> torch.optim.Optimizer:
        pass

    def _validate_config(self, config: Dict) -> None:
        required = {"num_classes", "classifier"}
        if not required.issubset(config.keys()):
            missing = required - config.keys()
            raise ModelConfigurationError(f"Missing config keys: {missing}")
