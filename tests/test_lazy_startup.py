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
    def setUp(self):
        EmbeddingManager._reset_singleton()

    def test_import_and_health_do_not_load_embedding_model(self):
        """Verifies that importing main and querying /health does not eagerly instantiate ONNX model."""
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

        mock_session = Mock()
        mock_tokenizer = Mock()

        with (
            patch("huggingface_hub.hf_hub_download", return_value="/mock/path"),
            patch("tokenizers.Tokenizer.from_file", return_value=mock_tokenizer),
            patch("onnxruntime.InferenceSession", return_value=mock_session),
        ):
            self.assertFalse(test_mgr.is_loaded())
            test_mgr._ensure_model(local_files_only=True)
            self.assertTrue(test_mgr.is_loaded())


if __name__ == "__main__":
    unittest.main()
