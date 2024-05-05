from pydantic import BaseModel


class DetailResponseBody(BaseModel):
    detail: str


NO_SEGMENTATION = {
    404: {
        "description": "Error: Could not segment image",
        "model": DetailResponseBody,
    }
}
