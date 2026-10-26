import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend import main
from backend.embedding import EmbeddingManager, embedding_manager


class LazyStartupAndHealthTests(unittest.TestCase):
    def test_import_and_health_do_not_load_embedding_model(self):
        """Verifies that importing main and querying /health does not eagerly instantiate SentenceTransformer."""
        # Fresh isolated manager
        manager = EmbeddingManager()
        self.assertFalse(manager.is_loaded())

        with (
            patch.object(main, "LLM_PROVIDER", "groq"),
            patch.object(main.llm_client, "check_health", return_value=(True, "Groq is online.")),
            patch.object(main.vector_store, "get_current_document", return_value=None),
            patch.object(main.vector_store, "has_active_document", return_value=False),
        ):
            response = main.health_check()
            payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "healthy")
        self.assertFalse(manager.is_loaded())

    def test_lazy_loading_only_triggers_on_encoding_call(self):
        """Verifies that _ensure_model is deferred until encode_texts or encode_query is called."""
        test_mgr = EmbeddingManager()
        # Mock SentenceTransformer so no real weights or network calls are made
        mock_st = Mock()
        mock_st.return_value.encode.return_value = Mock(astype=lambda x: [0.1, 0.2])

        with patch("sentence_transformers.SentenceTransformer", mock_st):
            # Model should still not be loaded before call
            self.assertFalse(test_mgr.is_loaded())
            # Encoding triggers lazy loading
            test_mgr._ensure_model()
            self.assertTrue(test_mgr.is_loaded())


if __name__ == "__main__":
    unittest.main()
