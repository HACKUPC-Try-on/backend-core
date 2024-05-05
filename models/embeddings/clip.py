from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from models.embeddings.embedding_model import EmbeddingModel


class CLIP(EmbeddingModel):
    model: CLIPModel
    processor: CLIPProcessor

    def __init__(self) -> None:
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # type: ignore
        self.model = self.model.to(self.device)  # type: ignore
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )  # type: ignore
        self.output_size = 512

    def get_image_features(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        embeddings = self.model.get_image_features(**inputs.to(self.device))  # type: ignore
        embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings
