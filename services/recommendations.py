from pathlib import Path
from fastapi import HTTPException, UploadFile

from models.faiss import FaissIndex
from services.image import ImageService
from services.segment import SegmentService


class RecommendationsService:
    image_service: ImageService
    faiss: FaissIndex
    segment_service: SegmentService

    def __init__(self) -> None:
        self.image_service = ImageService()
        self.faiss = FaissIndex()
        self.segment_service = SegmentService()

    def get_recommendations(self, image: UploadFile, category: str):
        image_file = self.image_service.save_upload_image(image)
        image_path = f"static/upload/{image_file}"
        path_image = Path(image_path)
        img = self.segment_service.segment_item(path_image, [category])
        if img is None:
            raise HTTPException(
                status_code=404, detail="Could not segment image"
            )
        return self.faiss.search_image(img)
