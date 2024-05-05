from fastapi import APIRouter, status

from services.index import IndexService


router = APIRouter(prefix="/index", tags=["index"])

service = IndexService()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Genereate index",
    description="Genearte faiss index with existing images",
)
def generate_index():
    return service.generate_index()
