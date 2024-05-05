from models.faiss import FaissIndex


class IndexService:
    @staticmethod
    def generate_index() -> None:
        faiss = FaissIndex()
        faiss.generate_index()
