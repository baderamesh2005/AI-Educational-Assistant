import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from backend.config import VECTOR_STORE_DIR

logger = logging.getLogger(__name__)


class InterProcessLock:
    """
    Cross-platform file lock for coordinating file writes across processes
    using msvcrt on Windows and fcntl on Unix/Linux.
    """
    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.fd = None

    def acquire(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_file, "a+")
        if sys.platform == "win32":
            import msvcrt
            self.fd.seek(0)
            msvcrt.locking(self.fd.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)

    def release(self):
        if self.fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.fd.seek(0)
                    msvcrt.locking(self.fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                self.fd.close()
                self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VectorStoreManager:
    """
    Manages vector indexing and retrieval with generation-based atomic updates,
    validation checks, thread/process synchronization, and backward-compatible legacy loading.
    """
    _instance = None

    def __new__(cls, storage_dir: Optional[Path] = None):
        # Default global singleton uses VECTOR_STORE_DIR
        if storage_dir is None:
            if cls._instance is None:
                instance = super(VectorStoreManager, cls).__new__(cls)
                instance._setup(VECTOR_STORE_DIR)
                cls._instance = instance
            return cls._instance
        # Isolated non-singleton instance for tests
        instance = super(VectorStoreManager, cls).__new__(cls)
        instance._setup(storage_dir)
        return instance

    @classmethod
    def create_isolated(cls, storage_dir: Path) -> "VectorStoreManager":
        """Creates an independent VectorStoreManager isolated from the global singleton."""
        instance = super(VectorStoreManager, cls).__new__(cls)
        instance._setup(storage_dir)
        return instance

    def _setup(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.generations_dir = self.storage_dir / "generations"
        self.manifest_path = self.storage_dir / "active_manifest.json"
        self.lock_path = self.storage_dir / ".write.lock"
        self._lock = threading.RLock()

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.current_doc: Optional[Dict[str, Any]] = None
        self.active_manifest: Optional[Dict[str, Any]] = None

        self._init_from_disk()

    def _init_from_disk(self):
        """
        Loads index and metadata into memory on startup.
        Prioritizes validated generation via active_manifest.json.
        Falls back to legacy files (current.index, current_metadata.json, current_document.json)
        if manifest is absent or invalid.
        """
        with self._lock:
            # 1. Try active_manifest.json
            if self.manifest_path.exists():
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)

                    rel_path = manifest.get("relative_path")
                    if rel_path:
                        gen_dir = self.storage_dir / rel_path
                        idx_file = gen_dir / "index.faiss"
                        meta_file = gen_dir / "metadata.json"
                        doc_file = gen_dir / "document.json"

                        if idx_file.exists() and meta_file.exists() and doc_file.exists():
                            idx = faiss.read_index(str(idx_file))
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            with open(doc_file, "r", encoding="utf-8") as f:
                                doc = json.load(f)

                            # Integrity validation
                            if (
                                isinstance(meta, list)
                                and len(meta) > 0
                                and idx.ntotal == len(meta)
                                and isinstance(doc, dict)
                            ):
                                self.index = idx
                                self.metadata = meta
                                self.current_doc = doc
                                self.active_manifest = manifest
                                logger.info(
                                    "[VectorStoreManager] Loaded generation '%s' (%d vectors for '%s')",
                                    manifest.get("generation_id"), idx.ntotal, doc.get("filename")
                                )
                                return
                            else:
                                logger.warning(
                                    "[VectorStoreManager] Generation '%s' failed integrity check (vectors=%d, meta=%s). Falling back.",
                                    manifest.get("generation_id"), idx.ntotal, len(meta) if isinstance(meta, list) else "invalid"
                                )
                except Exception as e:
                    logger.warning("[VectorStoreManager] Error loading manifest generation: %s. Falling back.", e)

            # 2. Fallback to legacy files
            legacy_idx = self.storage_dir / "current.index"
            legacy_meta = self.storage_dir / "current_metadata.json"
            legacy_doc = self.storage_dir / "current_document.json"

            if legacy_idx.exists() and legacy_meta.exists() and legacy_doc.exists():
                try:
                    idx = faiss.read_index(str(legacy_idx))
                    with open(legacy_meta, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    with open(legacy_doc, "r", encoding="utf-8") as f:
                        doc = json.load(f)

                    # Legacy integrity check
                    if (
                        isinstance(meta, list)
                        and len(meta) > 0
                        and idx.ntotal == len(meta)
                        and isinstance(doc, dict)
                    ):
                        self.index = idx
                        self.metadata = meta
                        self.current_doc = doc
                        self.active_manifest = None
                        logger.info(
                            "[VectorStoreManager] Loaded legacy index with %d vectors for '%s'",
                            idx.ntotal, doc.get("filename")
                        )
                        return
                    else:
                        logger.warning(
                            "[VectorStoreManager] Legacy index failed integrity check (vectors=%d, meta=%s).",
                            idx.ntotal, len(meta) if isinstance(meta, list) else "invalid"
                        )
                except Exception as e:
                    logger.warning("[VectorStoreManager] Error loading legacy index: %s", e)

            # 3. Clean empty state
            self.index = None
            self.metadata = []
            self.current_doc = None
            self.active_manifest = None

    def has_active_document(self) -> bool:
        with self._lock:
            return self.index is not None and len(self.metadata) > 0 and self.current_doc is not None

    def get_current_document(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.current_doc

    def get_active_manifest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.active_manifest

    def replace_with_new_document(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        filename: str,
        total_pages: int
    ) -> Dict[str, Any]:
        """
        Safely and atomically replaces the active FAISS index, metadata, and document info.
        Builds and validates a new generation directory before activating via manifest.
        Preserves existing valid generation if an error occurs.
        """
        if not chunks:
            raise ValueError("Cannot index 0 chunks.")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})."
            )

        dimension = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(embeddings)

        doc_info = {
            "filename": filename,
            "pages": total_pages,
            "chunks": len(chunks),
            "status": "ready"
        }

        gen_id = f"gen_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        gen_dir = self.generations_dir / gen_id

        with self._lock:
            with InterProcessLock(self.lock_path):
                gen_dir.mkdir(parents=True, exist_ok=True)
                idx_file = gen_dir / "index.faiss"
                meta_file = gen_dir / "metadata.json"
                doc_file = gen_dir / "document.json"

                # 1. Write index
                faiss.write_index(new_index, str(idx_file))

                # 2. Write metadata atomically
                meta_tmp = meta_file.with_suffix(".json.tmp")
                with open(meta_tmp, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                meta_tmp.replace(meta_file)

                # 3. Write document info atomically
                doc_tmp = doc_file.with_suffix(".json.tmp")
                with open(doc_tmp, "w", encoding="utf-8") as f:
                    json.dump(doc_info, f, ensure_ascii=False, indent=2)
                doc_tmp.replace(doc_file)

                # 4. Strict validation of completed generation
                validated_idx = faiss.read_index(str(idx_file))
                if validated_idx.ntotal != len(chunks):
                    raise RuntimeError(
                        f"Validation error: written vectors ({validated_idx.ntotal}) != expected chunks ({len(chunks)})"
                    )

                with open(meta_file, "r", encoding="utf-8") as f:
                    validated_meta = json.load(f)
                if len(validated_meta) != len(chunks):
                    raise RuntimeError(
                        f"Validation error: written metadata items ({len(validated_meta)}) != expected ({len(chunks)})"
                    )

                # 5. Activate by atomic manifest replacement
                manifest_data = {
                    "generation_id": gen_id,
                    "relative_path": f"generations/{gen_id}",
                    "filename": filename,
                    "total_pages": total_pages,
                    "chunks": len(chunks),
                    "vectors": new_index.ntotal,
                    "created_at": time.time(),
                }
                manifest_tmp = self.manifest_path.with_suffix(f".json.tmp.{os.getpid()}")
                with open(manifest_tmp, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, ensure_ascii=False, indent=2)
                manifest_tmp.replace(self.manifest_path)

                # 6. Mirror legacy files atomically for backward compatibility
                try:
                    legacy_idx = self.storage_dir / "current.index"
                    legacy_meta = self.storage_dir / "current_metadata.json"
                    legacy_doc = self.storage_dir / "current_document.json"

                    legacy_idx_tmp = legacy_idx.with_suffix(".index.tmp")
                    faiss.write_index(new_index, str(legacy_idx_tmp))
                    legacy_idx_tmp.replace(legacy_idx)

                    legacy_meta_tmp = legacy_meta.with_suffix(".json.tmp")
                    with open(legacy_meta_tmp, "w", encoding="utf-8") as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    legacy_meta_tmp.replace(legacy_meta)

                    legacy_doc_tmp = legacy_doc.with_suffix(".json.tmp")
                    with open(legacy_doc_tmp, "w", encoding="utf-8") as f:
                        json.dump(doc_info, f, ensure_ascii=False, indent=2)
                    legacy_doc_tmp.replace(legacy_doc)
                except Exception as legacy_err:
                    logger.warning("[VectorStoreManager] Failed to mirror legacy files: %s", legacy_err)

                # 7. Update in-memory state
                self.index = new_index
                self.metadata = chunks
                self.current_doc = doc_info
                self.active_manifest = manifest_data

                logger.info(
                    "[VectorStoreManager] Activated generation '%s' for '%s' (%d chunks, %d vectors).",
                    gen_id, filename, len(chunks), new_index.ntotal
                )
                return doc_info

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches the in-memory FAISS index and returns matched chunks with similarity scores.
        """
        if not self.has_active_document():
            raise RuntimeError("No active document indexed. Please upload a textbook PDF first.")

        with self._lock:
            actual_k = min(top_k, self.index.ntotal)
            if actual_k <= 0:
                return []

            distances, indices = self.index.search(query_embedding, actual_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self.metadata):
                    continue
                chunk_data = dict(self.metadata[idx])
                chunk_data["similarity"] = round(float(dist), 4)
                results.append(chunk_data)

            return results


# Global singleton
vector_store = VectorStoreManager()
