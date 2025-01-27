# data/transforms/base_transforms.py
from abc import ABC, abstractmethod
from torchvision import transforms


class BaseTransformationStrategy(ABC):
    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        pass


class TrainingTransformationStrategy(BaseTransformationStrategy):
    def __init__(self, img_size: int, mean: list, std: list):
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def get_transforms(self) -> transforms.Compose:
        return self.transform


class ValidationTransformationStrategy(BaseTransformationStrategy):
    def __init__(self, img_size: int, mean: list, std: list):
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def get_transforms(self) -> transforms.Compose:
        return self.transform
