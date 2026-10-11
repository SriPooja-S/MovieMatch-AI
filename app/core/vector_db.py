import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.utils.logger import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("[VectorDB] FAISS not found — using NumPy memory-mapped engine")


class VectorStore:
    """
    CPU-Optimized FAISS / Memory-Mapped Vector Search Engine.
    Uses Inner Product (IP) index on L2-normalized vectors for exact cosine similarity search.
    Supports multi-collection indexing: 'movie_plots' and 'movie_frames'.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self._is_faiss = FAISS_AVAILABLE

    def build_index(self, vectors: np.ndarray):
        """Build FAISS index or memory-mapped vector array."""
        if vectors is None or len(vectors) == 0:
            logger.warning("[VectorDB] Empty vector matrix provided to build_index")
            return

        # Ensure Float32 and L2 normalization
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        self.dimension = vectors.shape[1]
        self.vectors = vectors

        if self._is_faiss:
            try:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(vectors)
                logger.info(f"[VectorDB] Built FAISS-CPU IndexFlatIP with {self.index.ntotal} vectors (Dim: {self.dimension})")
            except Exception as e:
                logger.error(f"[VectorDB] Failed to build FAISS index: {e}. Falling back to NumPy.")
                self._is_faiss = False
        else:
            logger.info(f"[VectorDB] Loaded NumPy matrix with {len(vectors)} vectors (Dim: {self.dimension})")

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search vector index.
        Returns tuple of (scores_percentage, indices).
        """
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([]), np.array([])

        query = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if self._is_faiss and self.index is not None:
            scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
            scores_pct = scores[0] * 100.0
            return scores_pct, indices[0]
        else:
            # NumPy dot product
            scores = np.dot(self.vectors, query.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            scores_pct = scores[top_indices] * 100.0
            return scores_pct, top_indices
