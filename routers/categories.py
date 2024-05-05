from fastapi import APIRouter, File, HTTPException, UploadFile, status

from schemas.errors import NO_SEGMENTATION
from services.categories import CategoryService

router = APIRouter(prefix="/categories", tags=["categories"])

service = CategoryService()


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Get categories",
    description="Get all possible categories from an image",
    response_model=list[str],
    responses={**NO_SEGMENTATION},
)
def get_items(image: UploadFile = File(...)):
    items = service.get_items(image)
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No segmentation found",
        )
    return items
