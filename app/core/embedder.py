import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.utils.config import settings
from app.utils.logger import logger


class EmbeddingGenerator:
    """
    Generate vector embeddings for text using sentence transformers.
    Uses a multilingual-capable model for global movie support.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        batch_size = batch_size or settings.BATCH_SIZE
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2-normalize for better cosine sim
        )
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate and normalize embedding for a single text."""
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
