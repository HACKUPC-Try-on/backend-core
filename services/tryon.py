from fastapi import UploadFile
from PIL import Image
import io
import requests

from services.image import ImageService


class TryOnService:
    image_service: ImageService

    def __init__(self):
        self.image_service = ImageService()

    def try_on(self, background: UploadFile, cloth: UploadFile) -> str:
        human_img_path = self.image_service.save_upload_image(background)
        cloth_img_path = self.image_service.save_upload_image(cloth)

        human_img_bytes = self._get_file_from_image(
            f"static/upload/{human_img_path}", "human_img"
        )
        cloth_img_bytes = self._get_file_from_image(
            f"static/upload/{cloth_img_path}", "cloth_img"
        )

        files = self._get_multiple_files([human_img_bytes, cloth_img_bytes])
        print("About to make backend request")
        response = requests.post(self._get_api_endpoint(), files=files)

        return self.save_byte_img(response.content)

    def _get_api_endpoint(self) -> str:
        return "http://79.116.40.166:32490/try-on/"

    def _get_file_from_image(
        self, image_path: str, label: str
    ) -> tuple[str, tuple[str, bytes, str]]:
        img = Image.open(image_path)
        return label, ("upload.jpg", self._img_to_bytes(img), "image/jpeg,")

    def _img_to_bytes(self, img: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.read()

    def _get_multiple_files(
        self, files: list[tuple[str, tuple[str, bytes, str]]]
    ) -> dict[str, tuple[str, bytes, str]]:
        return {
            label: (filename, img_bytes, content_type)
            for label, (filename, img_bytes, content_type) in files
        }

    def save_byte_img(self, img: bytes) -> str:
        tryon_path = "static/tryon/tryon.jpg"
        pil_img = Image.open(io.BytesIO(img))
        pil_img.save(tryon_path)
        return tryon_path
