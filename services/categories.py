from fastapi import UploadFile
from models.grounding_dino import GroundingDINO
from services.image import ImageService
from models.grounding_dino import DEFAULT_CLASSES


class CategoryService:
    dino: GroundingDINO
    img_service: ImageService
    confidence: float

    def __init__(self) -> None:
        self.dino = GroundingDINO()
        self.img_service = ImageService()
        self.confidence = 0.3

    def get_items(self, image: UploadFile):
        image_file = self.img_service.save_upload_image(image)
        image_path = f"static/upload/{image_file}"
        predictions = self.dino.predict(image_path)
        items: list[str] = []
        for _, _, confidence, class_id, _, _ in predictions:
            print(
                f"Detected class: {DEFAULT_CLASSES[class_id]} with confidence: {confidence}"
            )
            if confidence > self.confidence:
                items.append(DEFAULT_CLASSES[class_id])
        return sorted(set(items))
