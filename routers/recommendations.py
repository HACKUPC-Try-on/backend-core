from fastapi import APIRouter, File, UploadFile, status

from services.recommendations import RecommendationsService
from schemas.errors import NO_SEGMENTATION

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

service = RecommendationsService()


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    summary="Get recommendations",
    description="Get recommendations from an image",
    response_model=list[str],
    responses={**NO_SEGMENTATION},
)
def get_recommendations(
    category: str,
    image: UploadFile = File(...),
):
    a = service.get_recommendations(image, category)
    print(a)
    return a
