from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
import numpy as np


class SegmentAnythingMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SegmentAnything(metaclass=SegmentAnythingMeta):
    model: SamPredictor

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "weights/sam/sam_vit_h_4b8939.pth"
        encoder_version = "vit_h"
        model_registry = sam_model_registry[encoder_version](
            checkpoint=checkpoint_path
        ).to(device)
        self.model = SamPredictor(model_registry)

    def segment(self, image_path: str, xyxy: np.ndarray) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.model.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, _ = self.model.predict(
                box=box,
                multimask_output=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
