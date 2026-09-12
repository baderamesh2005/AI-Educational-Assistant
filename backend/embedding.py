import time
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL_NAME

class EmbeddingManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            print(f"[EmbeddingManager] Initializing and loading {EMBEDDING_MODEL_NAME} into memory...")
            start_t = time.time()
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            load_dur = time.time() - start_t
            print(f"[EmbeddingManager] Loaded {EMBEDDING_MODEL_NAME} in {load_dur:.2f}s (Kept resident in memory).")
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized embeddings for a list of text chunks.
        L2-normalization enables exact cosine similarity using Inner Product in FAISS.
        """
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
        embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype("float32")

# Global singleton accessor
embedding_manager = EmbeddingManager()
