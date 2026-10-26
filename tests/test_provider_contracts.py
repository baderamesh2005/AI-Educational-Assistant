import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend import main
from backend.llm import OllamaClient, ProviderUnavailableError
from backend.mcq import MCQGenerationError, MCQGenerator


def valid_question(question_id=1):
    return {
        "id": question_id,
        "question": f"Which statement describes concept {question_id}?",
        "options": {
            "A": "First textbook-supported statement",
            "B": "Second distinct statement",
            "C": "Third distinct statement",
            "D": "Fourth distinct statement",
        },
        "correct_answer": "A",
        "explanation": "This is supported by the retrieved textbook context.",
        "page": 2,
    }


class ProviderContractTests(unittest.TestCase):
    def test_root_reports_active_groq_provider_and_model(self):
        with patch.object(main, "LLM_PROVIDER", "groq"), patch.object(
            main, "ACTIVE_MODEL_NAME", "openai/gpt-oss-120b"
        ):
            payload = json.loads(main.root().body)

        self.assertEqual(payload["provider"], "groq")
        self.assertEqual(payload["model"], "openai/gpt-oss-120b")

    def test_health_reports_configured_provider_failure(self):
        with (
            patch.object(main, "LLM_PROVIDER", "groq"),
            patch.object(main, "GROQ_MODEL_NAME", "openai/gpt-oss-120b"),
            patch.object(main.llm_client, "check_health", return_value=(False, "unavailable")),
            patch.object(main.vector_store, "get_current_document", return_value={"filename": "book.pdf"}),
            patch.object(main.vector_store, "has_active_document", return_value=True),
        ):
            payload = json.loads(main.health_check().body)

        self.assertEqual(payload["status"], "warning")
        self.assertEqual(payload["llm"]["provider"], "groq")
        self.assertEqual(payload["llm"]["model"], "openai/gpt-oss-120b")
        self.assertFalse(payload["llm"]["online"])
        self.assertFalse(payload["ollama"]["online"])
        self.assertTrue(payload["vector_store"]["has_active_doc"])

    def test_groq_health_requires_configured_model_to_be_available(self):
        client = OllamaClient()
        client.provider = "groq"
        client.groq_client = Mock()
        client.groq_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="openai/gpt-oss-120b")]
        )

        self.assertTrue(client.check_health()[0])

        client.groq_client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="another-model")]
        )
        online, message = client.check_health()
        self.assertFalse(online)
        self.assertNotIn("GROQ_API_KEY", message)

    def test_stream_provider_failure_emits_error_without_done(self):
        async def collect(body_iterator):
            return [chunk async for chunk in body_iterator]

        chunks = [{"page": 2, "chunk_id": 1, "text": "Relevant context."}]
        metrics = {
            "is_grounded": True,
            "confidence_pct": 80,
            "source_match_pct": 75,
        }
        with (
            patch.object(main.vector_store, "has_active_document", return_value=True),
            patch.object(main.retriever, "retrieve", return_value=(chunks, metrics)),
            patch.object(
                main.llm_client,
                "generate_answer_stream",
                side_effect=ProviderUnavailableError("private diagnostic"),
            ),
        ):
            response = main.ask_question_stream(main.QuestionRequest(question="Explain it"))
            events = asyncio.run(collect(response.body_iterator))

        payloads = [
            json.loads((chunk.decode() if isinstance(chunk, bytes) else chunk).split("data: ", 1)[1])
            for chunk in events
        ]
        self.assertEqual([event["type"] for event in payloads], ["meta", "error"])
        self.assertNotIn("private diagnostic", payloads[-1]["message"])

    def test_mcq_parser_rejects_incomplete_generated_set(self):
        generator = MCQGenerator()
        generator.provider = "groq"
        response = Mock()
        response.choices = [
            Mock(message=Mock(content=json.dumps({"mcqs": [valid_question()]})))
        ]
        generator.groq_client = Mock()
        generator.groq_client.chat.completions.create.return_value = response

        with self.assertRaises(MCQGenerationError):
            generator.generate_mcqs("textbook topic", [{"page": 2, "text": "Context"}])

    def test_mcq_parser_drops_questions_with_invalid_answers(self):
        generator = MCQGenerator()
        question = valid_question()
        question["correct_answer"] = "E"

        parsed = generator.clean_and_parse_json(
            json.dumps({"mcqs": [question]}),
            [2],
        )

        self.assertEqual(parsed, [])

    def test_mcq_generator_preserves_valid_five_question_shape(self):
        generator = MCQGenerator()
        generator.provider = "groq"
        response = Mock()
        response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {"mcqs": [valid_question(question_id) for question_id in range(1, 6)]}
                    )
                )
            )
        ]
        generator.groq_client = Mock()
        generator.groq_client.chat.completions.create.return_value = response

        mcqs, elapsed = generator.generate_mcqs(
            "textbook topic", [{"page": 2, "text": "Context"}]
        )

        self.assertEqual(len(mcqs), 5)
        self.assertTrue(all(mcq["correct_answer"] in mcq["options"] for mcq in mcqs))
        self.assertGreaterEqual(elapsed, 0)

    def test_mcq_parser_rejects_duplicate_questions(self):
        generator = MCQGenerator()
        repeated = valid_question()
        questions = [dict(repeated, id=question_id) for question_id in range(1, 6)]

        parsed = generator.clean_and_parse_json(
            json.dumps({"mcqs": questions}),
            [2],
        )

        self.assertEqual(len(parsed), 1)

    def test_ask_translates_provider_failure_to_safe_503(self):
        with (
            patch.object(main.vector_store, "has_active_document", return_value=True),
            patch.object(
                main.retriever,
                "retrieve",
                return_value=([{"page": 1, "text": "Textbook context"}], {"is_grounded": True}),
            ),
            patch.object(
                main.llm_client,
                "generate_answer_sync",
                side_effect=ProviderUnavailableError("private diagnostic"),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.ask_question(main.QuestionRequest(question="Explain this"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("private diagnostic", raised.exception.detail)

    def test_mcq_endpoint_translates_provider_failure_to_safe_503(self):
        with (
            patch.object(main.vector_store, "has_active_document", return_value=True),
            patch.object(
                main.retriever,
                "retrieve",
                return_value=([{"page": 1, "text": "Textbook context"}], {}),
            ),
            patch.object(
                main.mcq_generator,
                "generate_mcqs",
                side_effect=ProviderUnavailableError("private diagnostic"),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.generate_mcqs_endpoint(main.MCQRequest(topic="A textbook topic"))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("private diagnostic", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()