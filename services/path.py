from pathlib import Path
from models.faiss import DATA_PATH


class PathService:
    @staticmethod
    def get_embedding_path(image_id: str) -> Path:
        return Path(f"{DATA_PATH}{image_id}")

    @staticmethod
    def get_first_child(path: Path) -> Path:
        return next(path.iterdir())
