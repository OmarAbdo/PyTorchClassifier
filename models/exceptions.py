# models/exceptions.py
class ModelRegistrationError(Exception):
    """Raised when model registration fails validation"""


class ModelFactoryError(Exception):
    """Raised when model instantiation fails"""
