from fastapi import UploadFile
from pathlib import Path
import uuid
import cv2
import numpy as np

UPLOAD_FOLDER = "static/upload"


class ImageService:
    def save_upload_image(self, image: UploadFile) -> str:
        return self.save_image_to(image, Path(UPLOAD_FOLDER))

    def save_image_to(self, image: UploadFile, img_path: Path) -> str:
        extension = ".jpeg" if image.content_type == "image/jpeg" else ".png"
        img_name = f"{uuid.uuid4()}{extension}"
        path = img_path / img_name
        with open(path, "wb") as buffer:
            buffer.write(image.file.read())
        return img_name

    def get_cutout_from(self, img_path: str, mask: np.ndarray) -> np.ndarray:
        img = cv2.imread(img_path)
        mask = mask.astype(np.uint8)
        cutout = cv2.bitwise_and(img, img, mask=mask)
        black_pixels_mask = (
            (cutout[:, :, 0] == 0)
            & (cutout[:, :, 1] == 0)
            & (cutout[:, :, 2] == 0)
        )
        cutout[black_pixels_mask] = [255, 255, 255]
        return cutout
