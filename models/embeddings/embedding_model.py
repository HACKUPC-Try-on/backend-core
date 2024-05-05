from abc import ABC, ABCMeta, abstractmethod
import numpy as np
import torch


class EmbeddingModelMeta(type):
    _instances = {}

    def __call(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(EmbeddingModelMeta, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CombinedMeta(EmbeddingModelMeta, ABCMeta):
    def __new__(mcls, name, bases, namespace):
        return super().__new__(mcls, name, bases, namespace)


class EmbeddingModel(ABC, metaclass=CombinedMeta):
    output_size: int
    device: torch.device

    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def get_image_features(self, image: np.ndarray) -> np.ndarray:
        pass
