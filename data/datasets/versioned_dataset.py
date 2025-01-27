# data/datasets/versioned_dataset.py
import os
from typing import Tuple
from torchvision.datasets import ImageFolder
from config import AppConfig


class DatasetVersionError(Exception):
    """Custom exception for dataset version issues"""


class VersionedAnimalDataset(ImageFolder):
    def __init__(self, version: str = "v1", stage: str = "train"):
        config = AppConfig()
        data_config = config.get(f"data.versions.{version}", {})

        self._validate_version(data_config)

        transform_strategy = self._get_transform_strategy(stage, data_config)
        root = os.path.join(config.get("data.root_dir"), version, stage)

        super().__init__(root=root, transform=transform_strategy.get_transforms())

    def _validate_version(self, config: dict) -> None:
        """Pydantic-style validation without external deps"""
        required_keys = {"img_size", "mean", "std", "train_split"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            raise DatasetVersionError(f"Missing config keys: {missing}")

    def _get_transform_strategy(
        self, stage: str, config: dict
    ) -> BaseTransformationStrategy:
        if stage == "train":
            return TrainingTransformationStrategy(
                config["img_size"], config["mean"], config["std"]
            )
        return ValidationTransformationStrategy(
            config["img_size"], config["mean"], config["std"]
        )

    @classmethod
    def get_available_versions(cls) -> list:
        """Factory method to discover dataset versions"""
        config = AppConfig()
        return list(config.get("data.versions", {}).keys())
