from fastapi import APIRouter, status
from fastapi.responses import FileResponse

from services.path import PathService


router = APIRouter(prefix="/image", tags=["image"])


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Get image",
    description="Get image from an id",
    response_class=FileResponse,
)
def get_image(image_id: str):
    embedding_path = PathService.get_embedding_path(image_id)
    first_pic = PathService.get_first_child(embedding_path)
    return FileResponse(first_pic)
