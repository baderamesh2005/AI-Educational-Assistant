import json
import os
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np

from backend.config import FAISS_INDEX_PATH, METADATA_PATH, CURRENT_DOC_PATH

class VectorStoreManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance.index: Optional[faiss.Index] = None
            cls._instance.metadata: List[Dict[str, Any]] = []
            cls._instance.current_doc: Optional[Dict[str, Any]] = None
            cls._instance._init_from_disk()
        return cls._instance

    def _init_from_disk(self):
        """Loads index and metadata into memory on startup if already present."""
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists() and CURRENT_DOC_PATH.exists():
            try:
                print("[VectorStoreManager] Found existing vector store on disk. Loading into RAM...")
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                with open(CURRENT_DOC_PATH, "r", encoding="utf-8") as f:
                    self.current_doc = json.load(f)
                print(f"[VectorStoreManager] Loaded index with {self.index.ntotal} vectors for '{self.current_doc.get('filename')}'")
            except Exception as e:
                print(f"[VectorStoreManager] Error loading existing index: {e}. Resetting memory state.")
                self.index = None
                self.metadata = []
                self.current_doc = None
        else:
            self.index = None
            self.metadata = []
            self.current_doc = None

    def has_active_document(self) -> bool:
        return self.index is not None and len(self.metadata) > 0 and self.current_doc is not None

    def get_current_document(self) -> Optional[Dict[str, Any]]:
        return self.current_doc

    def replace_with_new_document(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        filename: str,
        total_pages: int
    ) -> Dict[str, Any]:
        """
        Atomically replaces the active FAISS index, metadata, and document info.
        Older files are replaced on disk and in memory.
        """
        dimension = embeddings.shape[1]

        # Use IndexFlatIP for cosine similarity with normalized vectors
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(embeddings)

        doc_info = {
            "filename": filename,
            "pages": total_pages,
            "chunks": len(chunks),
            "status": "ready"
        }

        # Write to disk
        faiss.write_index(new_index, str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        with open(CURRENT_DOC_PATH, "w", encoding="utf-8") as f:
            json.dump(doc_info, f, ensure_ascii=False, indent=2)

        # Update in-memory state
        self.index = new_index
        self.metadata = chunks
        self.current_doc = doc_info

        print(f"[VectorStoreManager] Replaced active document with '{filename}': {len(chunks)} chunks, {new_index.ntotal} vectors.")
        return doc_info

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches the in-memory FAISS index and returns the matched chunks with similarity scores.
        """
        if not self.has_active_document():
            raise RuntimeError("No active document indexed. Please upload a textbook PDF first.")

        actual_k = min(top_k, self.index.ntotal)
        if actual_k <= 0:
            return []

        distances, indices = self.index.search(query_embedding, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk_data = dict(self.metadata[idx])
            # For normalized vectors with IndexFlatIP, distance is cosine similarity (-1.0 to 1.0)
            chunk_data["similarity"] = round(float(dist), 4)
            results.append(chunk_data)

        return results

# Global singleton
vector_store = VectorStoreManager()
