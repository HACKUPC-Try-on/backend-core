import numpy as np
from models.embeddings.embedding_model import EmbeddingModel
from transformers import AutoModel, AutoProcessor
import torch


class Siglip(EmbeddingModel):
    model: AutoModel
    processor: AutoProcessor
    device: str

    def __init__(self) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")  # type: ignore
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")  # type: ignore
        self.output_size = 768

    def get_image_features(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )  # type: ignore
        embeddings = self.model.get_image_features(**inputs.to(self.device))  # type: ignore
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings
