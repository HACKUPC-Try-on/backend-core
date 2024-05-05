from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse
from services.tryon import TryOnService


router = APIRouter(
    prefix="/tryon",
    tags=["tryon"],
)

service = TryOnService()


@router.post(
    "",
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    summary="Try on a piece of clothing",
    description="Make an image that makes you try on a piece of clothing",
)
def try_on(
    human: UploadFile = File(...),
    cloth: UploadFile = File(...),
):
    res_path = service.try_on(human, cloth)
    return FileResponse(res_path)
