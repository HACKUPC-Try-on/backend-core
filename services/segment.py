from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import shutil
from models.grounding_dino import GroundingDINO
from models.sam import SegmentAnything
from services.image import ImageService


class SegmentService:
    grounding_dino: GroundingDINO
    sam: SegmentAnything
    image_service: ImageService
    last_segmented_path: Path
    failed_seg_path: Path

    def __init__(self):
        self.grounding_dino = GroundingDINO()
        self.segment_anything = SegmentAnything()
        self.image_service = ImageService()
        self.last_segmented_path = Path("static/segmented/seg.jpg")
        self.failed_seg_path = Path("static/segmented/fail.jpg")

    def segment_item(
        self, label_path: Path, classes: list[str]
    ) -> np.ndarray | None:
        detections = self.grounding_dino.predict(str(label_path), classes)
        detections.mask = self.segment_anything.segment(str(label_path), detections.xyxy)  # type: ignore

        cutouts: list[np.ndarray] = [
            self.image_service.get_cutout_from(str(label_path), mask)
            for mask in detections.mask  # type: ignore
        ]

        if not cutouts:
            return None

        # Get GD bounding box and crop the image with ti
        pil_img = Image.fromarray(cutouts[0])
        pil_img = pil_img.crop(detections.xyxy.tolist()[0])
        pil_img.save("test.jpg")
        return np.array(pil_img)

    def save_segmented_image(self, img: np.ndarray | None) -> None:
        if img is None:
            shutil.copy(self.failed_seg_path, self.last_segmented_path)
            return
        cv2.imwrite(str(self.last_segmented_path), img)
