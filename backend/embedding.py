import logging
import os
import threading
import time
from typing import List, Optional
import numpy as np

from backend.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

HF_MODEL_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256


class EmbeddingManager:
    """
    Manages text and query embeddings using CPU-only ONNX Runtime with thread-safe lazy loading.
    Replaces PyTorch/SentenceTransformer with an ultra-lightweight ONNX runtime session (~180MB RAM vs 530MB+),
    preventing Out-Of-Memory (OOM) termination on memory-constrained hosts (e.g. Render Free 512MB).
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._session = None
            cls._instance._tokenizer = None
            cls._instance._lock = threading.Lock()
        return cls._instance

    @classmethod
    def _reset_singleton(cls):
        """Resets the singleton instance state for test isolation."""
        if cls._instance is not None:
            cls._instance._session = None
            cls._instance._tokenizer = None
        cls._instance = None

    def is_loaded(self) -> bool:
        """Returns True if the underlying ONNX model and tokenizer are loaded in memory."""
        return self._session is not None and self._tokenizer is not None

    def _ensure_model(self, local_files_only: Optional[bool] = None):
        """
        Thread-safe lazy loader for the ONNX model and tokenizer.
        Initialises on first actual encoding call to keep startup memory minimal.
        """
        if self._session is None or self._tokenizer is None:
            with self._lock:
                if self._session is None or self._tokenizer is None:
                    start_t = time.perf_counter()
                    logger.info("[EmbeddingManager] Initialising CPU ONNX embedding engine for %s...", HF_MODEL_REPO_ID)

                    import onnxruntime as ort
                    from tokenizers import Tokenizer
                    from huggingface_hub import hf_hub_download

                    # Check environment override for offline / local-only loading
                    if local_files_only is None:
                        local_files_only = os.getenv("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")

                    try:
                        # Attempt to resolve model and tokenizer from local cache or download if online
                        model_path = hf_hub_download(
                            repo_id=HF_MODEL_REPO_ID,
                            filename="onnx/model.onnx",
                            local_files_only=local_files_only
                        )
                        tokenizer_path = hf_hub_download(
                            repo_id=HF_MODEL_REPO_ID,
                            filename="tokenizer.json",
                            local_files_only=local_files_only
                        )
                    except Exception as e:
                        logger.error("[EmbeddingManager] Failed to locate ONNX model artifacts for %s: %s", HF_MODEL_REPO_ID, e)
                        raise RuntimeError(
                            f"Missing ONNX model artifacts for '{HF_MODEL_REPO_ID}'. "
                            f"Ensure 'onnx/model.onnx' and 'tokenizer.json' are cached or reachable: {e}"
                        ) from e

                    # Load fast Rust-based tokenizer matching SentenceTransformer pipeline
                    tokenizer = Tokenizer.from_file(tokenizer_path)
                    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)

                    # Configure lightweight single-threaded CPU ONNX session
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    session = ort.InferenceSession(
                        model_path,
                        sess_options=opts,
                        providers=["CPUExecutionProvider"]
                    )

                    self._tokenizer = tokenizer
                    self._session = session
                    dur = time.perf_counter() - start_t
                    logger.info("[EmbeddingManager] ONNX model ready in %.2fs (Resident RAM ~180MB).", dur)

    @property
    def model(self):
        """Backward-compatible property accessor; ensures model is loaded."""
        self._ensure_model()
        return self._session

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates normalized 384-dimensional embeddings for a list of text strings.
        Uses exact mean pooling and L2 normalization matching the original SentenceTransformer pipeline.
        Compatible with FAISS Inner Product (cosine similarity) index.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        self._ensure_model()

        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self._tokenizer.encode_batch(batch_texts)

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )

            # outputs[0] has shape [batch_size, sequence_length, EMBEDDING_DIM]
            last_hidden_state = outputs[0]

            # Mean pooling with attention mask
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            # L2 normalization for exact cosine similarity in FAISS
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            normalized = (pooled / norms).astype(np.float32)

            all_embeddings.append(normalized)

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a normalized 2D numpy array [1, 384].
        """
        return self.encode_texts([query], batch_size=1)


# Global singleton accessor
embedding_manager = EmbeddingManager()
