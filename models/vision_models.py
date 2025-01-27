# models/vision_models.py
from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import optim

from .base_model import BaseAnimalModel
from .model_factory import ModelRegistry


@ModelRegistry.register("resnet50")
class ResNetAnimalModel(BaseAnimalModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_model = tv_models.resnet50(
            weights="DEFAULT" if config.get("pretrained", True) else None
        )
        self._build_classifier(config["classifier"])
        self.freeze_layers(config.get("frozen_layers", []))
        self.config = config  # Store merged config

    def _build_classifier(self, classifier_config: Dict) -> None:
        layers = []
        in_features = self.base_model.fc.in_features

        for hidden_dim in classifier_config["hidden_dims"]:
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_config.get("dropout", 0.5)),
            ]
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, self.config["num_classes"]))
        self.classifier = nn.Sequential(*layers)
        self.base_model.fc = self.classifier

    def freeze_layers(self, layer_names: List[str]) -> None:
        for name, param in self.base_model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False

    def get_optimizer(self, params) -> optim.Optimizer:
        opt_config = self.config["optimizer"]
        optimizer_class = getattr(optim, opt_config["type"])
        return optimizer_class(
            params, **{k: v for k, v in opt_config.items() if k != "type"}
        )
