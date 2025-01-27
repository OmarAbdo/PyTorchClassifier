# services/inference_service.py
from abc import ABC, abstractmethod

class IInferenceService(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass
    
    @abstractmethod
    def predict(self, image) -> dict:
        pass