
import json
import logging
import time
from typing import List, Dict, Any, Generator, Tuple

import requests
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

from backend.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    TEMPERATURE,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)


class ProviderUnavailableError(ConnectionError):
    """Raised when the configured upstream LLM provider cannot be reached."""


class LLMGenerationError(RuntimeError):
    """Raised when the provider is reachable but generation fails."""


class OllamaClient:
    def __init__(self):
        self.generate_url = f"{OLLAMA_BASE_URL}/api/generate"
        self.tags_url = f"{OLLAMA_BASE_URL}/api/tags"
        self.provider = LLM_PROVIDER
        self.groq_client = (
            Groq(api_key=GROQ_API_KEY, timeout=OLLAMA_TIMEOUT)
            if self.provider == "groq" and GROQ_API_KEY
            else None
        )

    def check_health(self) -> Tuple[bool, str]:
        if self.provider == "groq":
            if not self.groq_client:
                return False, "Groq is selected but is not configured."
            try:
                models = self.groq_client.models.list()
                available_models = {
                    getattr(model, "id", "")
                    for model in getattr(models, "data", [])
                }
                if GROQ_MODEL_NAME not in available_models:
                    return False, "The configured Groq model is not available."
                return True, f"Groq is online with model '{GROQ_MODEL_NAME}'."
            except Exception as e:
                logger.warning("Groq health check failed (%s).", type(e).__name__)
                return False, "The configured AI provider is unavailable."

        if self.provider != "ollama":
            return False, "The configured AI provider is not supported."

        try:
            res = requests.get(self.tags_url, timeout=4)
            res.raise_for_status()
            models = [m.get("name", "") for m in res.json().get("models", [])]
            matched = OLLAMA_MODEL_NAME in models
            if matched:
                return True, f"Ollama is online with model '{OLLAMA_MODEL_NAME}'."
            return False, f"Ollama model '{OLLAMA_MODEL_NAME}' was not found."
        except Exception as e:
            logger.warning("Ollama health check failed (%s).", type(e).__name__)
            return False, "The configured AI provider is unavailable."

    def build_qa_prompt(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "").strip()
            if len(text) > 700:
                text = text[:700] + "..."
            context_parts.append(f"[Textbook Excerpt {idx}]:\n{text}")

        context_str = "\n\n".join(context_parts)

        return f"""You are an expert AI Educational Assistant and teacher preparing Class 12 students for board examinations.
Answer based ONLY on the textbook excerpts below.

1. Do not use outside facts.
2. If the excerpts are insufficient, respond exactly:
"I could not find sufficient information about this topic in the uploaded textbook."
3. Do not mention page numbers.
4. Adapt the answer structure to the question: define, explain, compare, describe, or solve numerical problems step by step.
5. Use a clear, exam-ready academic tone.

TEXTBOOK EXCERPTS:
{context_str}

STUDENT QUESTION:
{question}

EXAM-READY ANSWER:"""

    def generate_answer_sync(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        prompt = self.build_qa_prompt(question, retrieved_chunks)
        t0 = time.perf_counter()

        try:
            if self.provider == "groq":
                if not self.groq_client:
                    raise ProviderUnavailableError(
                        "The configured AI provider is unavailable."
                    )
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=550,
                )
                answer = response.choices[0].message.content or ""
            elif self.provider == "ollama":
                payload = {
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_thread": 8,
                        "num_predict": 550,
                        "top_p": 0.9,
                    },
                }
                resp = requests.post(
                    self.generate_url, json=payload, timeout=OLLAMA_TIMEOUT
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "").strip()
            else:
                raise LLMGenerationError("The configured AI provider is not supported.")
        except ProviderUnavailableError:
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                APIConnectionError, APITimeoutError) as e:
            logger.warning("LLM provider unavailable (%s).", type(e).__name__)
            raise ProviderUnavailableError(
                "The configured AI provider is unavailable."
            ) from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            logger.warning("Ollama request failed with status %s.", status_code)
            if status_code in {401, 403, 404, 429} or status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from e
            raise LLMGenerationError("The AI provider could not generate a response.") from e
        except APIStatusError as e:
            logger.warning("Groq request failed with status %s.", e.status_code)
            if e.status_code in {401, 403, 404, 429} or e.status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from e
            raise LLMGenerationError("The AI provider could not generate a response.") from e
        except Exception as e:
            logger.error("LLM generation failed (%s).", type(e).__name__)
            raise LLMGenerationError(
                "The AI provider could not generate a response."
            ) from e

        if not answer.strip():
            raise LLMGenerationError("The AI provider returned an empty response.")

        return answer.strip(), round(time.perf_counter() - t0, 4)

    def generate_answer_stream(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        prompt = self.build_qa_prompt(question, retrieved_chunks)

        try:
            if self.provider == "groq":
                if not self.groq_client:
                    raise ProviderUnavailableError(
                        "The configured AI provider is unavailable."
                    )

                stream = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=550,
                    stream=True,
                )
                for event in stream:
                    if event.choices:
                        token = event.choices[0].delta.content
                        if token:
                            yield token
            elif self.provider == "ollama":
                payload = {
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_thread": 8,
                        "num_predict": 550,
                        "top_p": 0.9,
                    },
                }
                with requests.post(
                    self.generate_url,
                    json=payload,
                    stream=True,
                    timeout=OLLAMA_TIMEOUT,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        token = item.get("response", "")
                        if token:
                            yield token
                        if item.get("done", False):
                            break
            else:
                raise LLMGenerationError("The configured AI provider is not supported.")
        except ProviderUnavailableError:
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                APIConnectionError, APITimeoutError) as e:
            logger.warning("LLM stream provider unavailable (%s).", type(e).__name__)
            raise ProviderUnavailableError(
                "The configured AI provider is unavailable."
            ) from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            logger.warning("Ollama stream failed with status %s.", status_code)
            if status_code in {401, 403, 404, 429} or status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from e
            raise LLMGenerationError("The AI provider could not generate a response.") from e
        except APIStatusError as e:
            logger.warning("Groq stream failed with status %s.", e.status_code)
            if e.status_code in {401, 403, 404, 429} or e.status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from e
            raise LLMGenerationError("The AI provider could not generate a response.") from e
        except Exception as e:
            logger.error("LLM stream failed (%s).", type(e).__name__)
            raise LLMGenerationError(
                "The AI provider could not generate a response."
            ) from e


llm_client = OllamaClient()