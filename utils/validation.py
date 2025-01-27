from models.exceptions import ModelFactoryError
from models.model_factory import ModelRegistry


def validate_model_architecture(architecture: str) -> None:
    available_models = ModelRegistry._registry.keys()
    if architecture not in available_models:
        raise ModelFactoryError(
            f"Architecture {architecture} not registered. Available: {available_models}"
        )
