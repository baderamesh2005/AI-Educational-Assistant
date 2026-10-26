import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import DEFAULT_FRONTEND_ORIGINS, get_allowed_origins
from backend.vector_store import VectorStoreManager


def make_synthetic_chunks(count: int, doc_name: str = "TestDoc.pdf"):
    return [
        {
            "chunk_id": i + 1,
            "page": (i // 2) + 1,
            "document_name": doc_name,
            "text": f"This is synthetic text chunk number {i + 1} for document {doc_name}."
        }
        for i in range(count)
    ]


def make_synthetic_embeddings(count: int, dimension: int = 16) -> np.ndarray:
    raw = np.random.randn(count, dimension).astype("float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.clip(norms, 1e-12, None)


class VectorStoreReliabilityTests(unittest.TestCase):
    def test_successful_save_activation_and_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            manager = VectorStoreManager.create_isolated(storage_dir)

            self.assertFalse(manager.has_active_document())
            self.assertIsNone(manager.get_current_document())

            chunks = make_synthetic_chunks(5, "Biology.pdf")
            embeddings = make_synthetic_embeddings(5, dimension=32)

            doc_info = manager.replace_with_new_document(
                chunks=chunks,
                embeddings=embeddings,
                filename="Biology.pdf",
                total_pages=3
            )

            self.assertEqual(doc_info["filename"], "Biology.pdf")
            self.assertEqual(doc_info["chunks"], 5)
            self.assertTrue(manager.has_active_document())

            manifest = manager.get_active_manifest()
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest["vectors"], 5)

            # Test reload in a completely fresh isolated manager instance
            fresh_manager = VectorStoreManager.create_isolated(storage_dir)
            self.assertTrue(fresh_manager.has_active_document())
            self.assertEqual(fresh_manager.get_current_document()["filename"], "Biology.pdf")
            self.assertEqual(len(fresh_manager.metadata), 5)
            self.assertEqual(fresh_manager.index.ntotal, 5)

            # Test search
            query_vec = make_synthetic_embeddings(1, dimension=32)
            results = fresh_manager.search(query_vec, top_k=3)
            self.assertEqual(len(results), 3)
            self.assertIn("similarity", results[0])

    def test_index_vector_count_vs_metadata_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            # Create a corrupted legacy vector store with 5 vectors but only 3 chunks
            idx = faiss.IndexFlatIP(16)
            idx.add(make_synthetic_embeddings(5, 16))
            faiss.write_index(idx, str(storage_dir / "current.index"))

            meta = make_synthetic_chunks(3, "Mismatched.pdf")
            with open(storage_dir / "current_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            with open(storage_dir / "current_document.json", "w", encoding="utf-8") as f:
                json.dump({"filename": "Mismatched.pdf", "pages": 2, "chunks": 3}, f)

            manager = VectorStoreManager.create_isolated(storage_dir)
            # Should safely reject mismatched data and stay inactive
            self.assertFalse(manager.has_active_document())
            self.assertIsNone(manager.index)
            self.assertEqual(manager.metadata, [])

    def test_invalid_or_incomplete_generation_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            manager = VectorStoreManager.create_isolated(storage_dir)

            # Write broken manifest pointing to non-existent generation
            manifest_file = storage_dir / "active_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump({
                    "generation_id": "gen_broken",
                    "relative_path": "generations/gen_broken",
                    "filename": "Broken.pdf"
                }, f)

            # Fresh manager reading broken manifest
            fresh_manager = VectorStoreManager.create_isolated(storage_dir)
            self.assertFalse(fresh_manager.has_active_document())

    def test_write_failure_preserves_previous_active_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            manager = VectorStoreManager.create_isolated(storage_dir)

            # 1. Establish valid Generation 1
            chunks_1 = make_synthetic_chunks(4, "FirstDoc.pdf")
            embeddings_1 = make_synthetic_embeddings(4, 16)
            manager.replace_with_new_document(chunks_1, embeddings_1, "FirstDoc.pdf", 2)

            first_manifest = manager.get_active_manifest()
            self.assertEqual(manager.get_current_document()["filename"], "FirstDoc.pdf")
            self.assertEqual(manager.index.ntotal, 4)

            # 2. Attempt Generation 2 with simulated error during index write
            chunks_2 = make_synthetic_chunks(6, "SecondDoc.pdf")
            embeddings_2 = make_synthetic_embeddings(6, 16)

            with patch("faiss.write_index", side_effect=IOError("Simulated disk write failure")):
                with self.assertRaises(IOError):
                    manager.replace_with_new_document(chunks_2, embeddings_2, "SecondDoc.pdf", 3)

            # 3. Assert previous generation remains active both in memory and on disk
            self.assertTrue(manager.has_active_document())
            self.assertEqual(manager.get_current_document()["filename"], "FirstDoc.pdf")
            self.assertEqual(manager.index.ntotal, 4)
            self.assertEqual(manager.get_active_manifest()["generation_id"], first_manifest["generation_id"])

            # Reload to confirm disk state is intact
            fresh_manager = VectorStoreManager.create_isolated(storage_dir)
            self.assertTrue(fresh_manager.has_active_document())
            self.assertEqual(fresh_manager.get_current_document()["filename"], "FirstDoc.pdf")

    def test_concurrent_write_attempts_safely_serialized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            manager = VectorStoreManager.create_isolated(storage_dir)

            def worker_write(doc_id: int):
                chunks = make_synthetic_chunks(3, f"Doc_{doc_id}.pdf")
                embeddings = make_synthetic_embeddings(3, 16)
                manager.replace_with_new_document(chunks, embeddings, f"Doc_{doc_id}.pdf", 1)
                return doc_id

            # Run 5 concurrent updates
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker_write, i) for i in range(5)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            self.assertEqual(len(results), 5)
            # The manager must be left in a consistent state
            self.assertTrue(manager.has_active_document())
            self.assertEqual(manager.index.ntotal, 3)
            self.assertEqual(len(manager.metadata), 3)

            # Fresh manager reload confirms on-disk consistency
            fresh_manager = VectorStoreManager.create_isolated(storage_dir)
            self.assertTrue(fresh_manager.has_active_document())
            self.assertEqual(fresh_manager.index.ntotal, 3)
            self.assertEqual(len(fresh_manager.metadata), 3)

    def test_loading_legacy_file_layout_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            # Construct legacy layout
            legacy_idx = storage_dir / "current.index"
            legacy_meta = storage_dir / "current_metadata.json"
            legacy_doc = storage_dir / "current_document.json"

            idx = faiss.IndexFlatIP(16)
            idx.add(make_synthetic_embeddings(4, 16))
            faiss.write_index(idx, str(legacy_idx))

            chunks = make_synthetic_chunks(4, "LegacyBook.pdf")
            with open(legacy_meta, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)

            doc_info = {"filename": "LegacyBook.pdf", "pages": 5, "chunks": 4, "status": "ready"}
            with open(legacy_doc, "w", encoding="utf-8") as f:
                json.dump(doc_info, f, indent=2)

            # Record mtimes
            idx_mtime_before = legacy_idx.stat().st_mtime_ns
            meta_mtime_before = legacy_meta.stat().st_mtime_ns

            manager = VectorStoreManager.create_isolated(storage_dir)
            self.assertTrue(manager.has_active_document())
            self.assertEqual(manager.get_current_document()["filename"], "LegacyBook.pdf")
            self.assertEqual(manager.index.ntotal, 4)
            self.assertEqual(len(manager.metadata), 4)

            # Ensure files were not overwritten or touched during read
            self.assertEqual(legacy_idx.stat().st_mtime_ns, idx_mtime_before)
            self.assertEqual(legacy_meta.stat().st_mtime_ns, meta_mtime_before)

            # Manifest was not forcefully written during legacy read
            self.assertFalse((storage_dir / "active_manifest.json").exists())

    def test_storage_isolation_through_app_storage_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["APP_STORAGE_DIR"] = temp_dir

            test_script = """
import numpy as np
import faiss
from backend import config
from backend.vector_store import VectorStoreManager

manager = VectorStoreManager.create_isolated(config.VECTOR_STORE_DIR)
chunks = [{"chunk_id": 1, "page": 1, "document_name": "Isolated.pdf", "text": "Isolated test"}]
embeddings = np.random.randn(1, 16).astype("float32")
manager.replace_with_new_document(chunks, embeddings, "Isolated.pdf", 1)

assert (config.VECTOR_STORE_DIR / "active_manifest.json").exists()
print("ISOLATION_OK")
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            self.assertIn("ISOLATION_OK", result.stdout)
            self.assertTrue((Path(temp_dir) / "vector_store" / "active_manifest.json").exists())


class FrontendOriginsNormalizationTests(unittest.TestCase):
    def test_default_origins_used_when_unset_or_blank(self):
        self.assertEqual(get_allowed_origins(""), DEFAULT_FRONTEND_ORIGINS)
        self.assertEqual(get_allowed_origins("   "), DEFAULT_FRONTEND_ORIGINS)
        self.assertEqual(get_allowed_origins(",,,"), DEFAULT_FRONTEND_ORIGINS)

    def test_trailing_slashes_and_whitespace_removed(self):
        raw = " https://app.vercel.app/ , http://localhost:3000/ "
        expected = ["https://app.vercel.app", "http://localhost:3000"]
        self.assertEqual(get_allowed_origins(raw), expected)

    def test_empty_segments_ignored(self):
        raw = "https://app1.vercel.app, , https://app2.vercel.app/"
        expected = ["https://app1.vercel.app", "https://app2.vercel.app"]
        self.assertEqual(get_allowed_origins(raw), expected)

    def test_valid_custom_origin_preserved(self):
        raw = "https://my-domain.org"
        self.assertEqual(get_allowed_origins(raw), ["https://my-domain.org"])


if __name__ == "__main__":
    unittest.main()
