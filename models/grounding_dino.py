import torch
import numpy as np
import cv2
from packages.GroundingDINO.groundingdino.util.inference import Model

DEFAULT_CLASSES = ["shirt", "pants", "bag", "shoes", "sweater"]


class GroundingDINOMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class GroundingDINO(metaclass=GroundingDINOMeta):
    model: Model

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = "weights/dino/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "weights/dino/groundingdino_swint_ogc.pth"
        self.model = Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=device,
        )

    def predict(
        self, image_path: str, classes: list[str] = DEFAULT_CLASSES
    ) -> list[str]:
        return self._predict(image_path, classes)

    def _predict(self, image_path: str, classes: list[str]) -> list[str]:
        box_threshold = 0.35
        text_threshold = 0.25

        image = self._read_image(image_path)
        detections = self.model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return detections

    def _read_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return image
