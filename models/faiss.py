from models.embeddings.clip import CLIP
from models.embeddings.embedding_model import EmbeddingModel
from models.embeddings.siglip import Siglip
from services.segment import SegmentService
import numpy as np
from pathlib import Path
import faiss
import csv
from models.grounding_dino import DEFAULT_CLASSES

INDEX_PATH = "static/indices/index.index"
LABEL_PATH = "static/indices/index.csv"
DATA_PATH = "static/embeddings/"


class FaissIndex:
    embedding_model: EmbeddingModel
    segment_service: SegmentService

    def __init__(self) -> None:
        self.embedding_model = Siglip()
        self.segment_service = SegmentService()

    def generate_index(self):
        index = faiss.IndexFlatL2(self.embedding_model.output_size)
        total_embeddings = 0
        with open(LABEL_PATH, "w") as label_file:
            writer = csv.writer(label_file)
            for image_dir in Path(DATA_PATH).iterdir():
                id = image_dir.stem
                print(f"Adding item {image_dir.stem} to the index")
                for img_path in image_dir.iterdir():
                    image = self.segment_service.segment_item(
                        img_path, DEFAULT_CLASSES
                    )
                    if image is None:
                        print(
                            f"Couldn't segment image {img_path}, continuing..."
                        )
                        continue
                    embedding = self.embedding_model.get_image_features(image)
                    index.add(embedding)  # type: ignore
                    writer.writerow([total_embeddings, id])
                    total_embeddings += 1
            faiss.write_index(index, INDEX_PATH)
            label_file.close()

    def search_image(self, img: np.ndarray) -> list[str]:
        results: list[str] = []
        embedding = self.embedding_model.get_image_features(img)
        faiss_index = faiss.read_index(INDEX_PATH)
        _, indices = faiss_index.search(embedding, 5)

        # Retrieve id and add distance and id to result
        reader = csv.reader(open(LABEL_PATH, "r"))
        for row in reader:
            index = int(row[0])
            if index in indices:
                results.append(row[1])
        return results
