# data/dataloaders/dataloader_factory.py
from torch.utils.data import DataLoader
from config import AppConfig


class DataLoaderFactory:
    def __init__(self):
        self.config = AppConfig()

    def create_dataloader(self, dataset, loader_type: str = "train") -> DataLoader:
        loader_config = self.config.get(f"data.{loader_type}", {})
        return DataLoader(
            dataset,
            batch_size=loader_config.get("batch_size", 32),
            shuffle=loader_config.get("shuffle", False),
            num_workers=loader_config.get("num_workers", 2),
            pin_memory=True,
        )

    @staticmethod
    def get_loader_types() -> list:
        """Returns available loader types from config"""
        return ["train", "validation", "test"]
