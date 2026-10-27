import os
import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.embedding import EmbeddingManager, EMBEDDING_DIM


class ONNXEmbeddingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force offline mode during unit tests to prevent unwanted external network calls
        os.environ["HF_HUB_OFFLINE"] = "1"
        cls.manager = EmbeddingManager()
        # Initialize locally from disk cache
        cls.manager._ensure_model(local_files_only=True)

    @classmethod
    def tearDownClass(cls):
        EmbeddingManager._reset_singleton()

    def test_output_shape_and_dtype(self):
        texts = [
            "Photosynthesis converts light energy into chemical energy in chloroplasts.",
            "Newton's laws of motion form the basis of classical mechanics."
        ]
        embeddings = self.manager.encode_texts(texts)
        self.assertEqual(embeddings.shape, (2, EMBEDDING_DIM))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_finite_values_and_normalization(self):
        texts = [
            "Thermodynamics studies heat, work, and energy transformations.",
            "DNA replication ensures genetic information is transferred accurately.",
            "",  # empty edge case
            "   "  # whitespace edge case
        ]
        embeddings = self.manager.encode_texts(texts)
        self.assertFalse(np.isnan(embeddings).any(), "Found NaN in embeddings")
        self.assertFalse(np.isinf(embeddings).any(), "Found Inf in embeddings")

        norms = np.linalg.norm(embeddings, axis=1)
        for i, norm in enumerate(norms):
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Vector {i} is not unit normalized: {norm}")

    def test_encode_query_single_vector(self):
        query = "What is the cell cycle?"
        query_vec = self.manager.encode_query(query)
        self.assertEqual(query_vec.shape, (1, EMBEDDING_DIM))
        self.assertEqual(query_vec.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(query_vec[0])), 1.0, places=5)

    def test_empty_input_handling(self):
        empty_res = self.manager.encode_texts([])
        self.assertEqual(empty_res.shape, (0, EMBEDDING_DIM))
        self.assertEqual(empty_res.dtype, np.float32)

    def test_batching_consistency(self):
        texts = [f"Educational concept number {i} for batching verification." for i in range(10)]
        batch_1 = self.manager.encode_texts(texts, batch_size=1)
        batch_5 = self.manager.encode_texts(texts, batch_size=5)
        np.testing.assert_allclose(batch_1, batch_5, atol=1e-5)

    def test_faiss_compatibility_and_ranking(self):
        import faiss

        documents = [
            "Mitochondria generate most of the chemical energy needed by eukaryotic cells.",
            "The French Revolution was a period of social and political upheaval in France.",
            "Gravitational potential energy depends on mass, gravity, and height."
        ]
        doc_embeddings = self.manager.encode_texts(documents)

        # Build FAISS Inner Product index (exact cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(doc_embeddings)
        self.assertEqual(index.ntotal, 3)

        # Query related to biology
        query = "Which organelle is responsible for cellular energy production?"
        query_vec = self.manager.encode_query(query)
        scores, indices = index.search(query_vec, k=1)

        self.assertEqual(indices[0][0], 0, "Top retrieved document should be mitochondria")
        self.assertGreater(scores[0][0], 0.4, "Cosine similarity score should be strong")

    def test_sentence_transformer_numerical_equivalence(self):
        """Validates that ONNX embeddings match SentenceTransformer to within 1e-4 absolute error and >0.9999 cosine similarity."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.skipTest("sentence-transformers not installed; skipping numerical equivalence check")

        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        test_sentences = [
            "Chemical equilibrium occurs when forward and reverse reactions proceed at equal rates.",
            "Light behaves both as a wave and as a stream of particles called photons.",
            "Mitosis results in two genetically identical daughter cells."
        ]

        st_embeddings = st_model.encode(test_sentences, normalize_embeddings=True)
        onnx_embeddings = self.manager.encode_texts(test_sentences)

        for i in range(len(test_sentences)):
            cos_sim = float(np.dot(st_embeddings[i], onnx_embeddings[i]))
            self.assertGreater(cos_sim, 0.9999, f"Sentence {i} cosine similarity below 0.9999: {cos_sim}")

        max_abs_diff = float(np.max(np.abs(st_embeddings - onnx_embeddings)))
        self.assertLess(max_abs_diff, 1e-4, f"Max absolute difference too large: {max_abs_diff}")


if __name__ == "__main__":
    unittest.main()
