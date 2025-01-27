# data/datasets/base_dataset.py
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseAnimalDataset(Dataset, ABC):
    @abstractmethod
    def get_class_labels(self) -> list:
        """Contract for label mapping"""
        pass
