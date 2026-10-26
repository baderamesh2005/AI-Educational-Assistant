import logging
import threading
import time
from typing import List, Optional
import numpy as np

from backend.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages text and query embeddings with thread-safe lazy loading.
    Defers importing SentenceTransformer and PyTorch until first actual encoding request,
    enabling fast startup and keeping initial memory consumption low for resource-constrained environments.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._lock = threading.Lock()
        return cls._instance

    def is_loaded(self) -> bool:
        """Returns True if the underlying model is already in memory without triggering a load."""
        return self._model is not None

    def _ensure_model(self):
        """Thread-safe lazy loader for the embedding model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("[EmbeddingManager] Lazy-loading %s into memory...", EMBEDDING_MODEL_NAME)
                    start_t = time.perf_counter()
                    from sentence_transformers import SentenceTransformer
                    import torch
                    # Conserve threads and memory pool allocations on single-core / low-RAM hosts
                    torch.set_num_threads(1)
                    self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                    dur = time.perf_counter() - start_t
                    logger.info("[EmbeddingManager] Loaded %s in %.2fs (Kept resident in memory).", EMBEDDING_MODEL_NAME, dur)

    @property
    def model(self):
        self._ensure_model()
        return self._model

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized embeddings for a list of text chunks.
        L2-normalization enables exact cosine similarity using Inner Product in FAISS.
        """
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a normalized 2D numpy array [1, dimension].
        """
        self._ensure_model()
        embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype("float32")


# Global singleton accessor
embedding_manager = EmbeddingManager()
