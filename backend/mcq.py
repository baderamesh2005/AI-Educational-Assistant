
import json
import logging
import re
import time
from typing import List, Dict, Any, Tuple

import requests
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

from backend.llm import LLMGenerationError, ProviderUnavailableError

from backend.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    MCQ_TEMPERATURE,
    OLLAMA_TIMEOUT,
)

logger = logging.getLogger(__name__)


class MCQGenerationError(ValueError):
    """Raised when a reliable textbook-grounded question set was not produced."""


class MCQGenerator:
    def __init__(self):
        self.generate_url = f"{OLLAMA_BASE_URL}/api/generate"
        self.provider = LLM_PROVIDER.lower()

        self.groq_client = (
            Groq(api_key=GROQ_API_KEY, timeout=OLLAMA_TIMEOUT)
            if self.provider == "groq" and GROQ_API_KEY
            else None
        )

    def build_mcq_prompt(
        self,
        topic: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        context_parts = []
        valid_pages = set()

        for idx, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("page", 1)
            valid_pages.add(page)

            text = chunk.get("text", "").strip()
            if len(text) > 500:
                text = text[:500] + "..."

            context_parts.append(
                f"[Textbook Content {idx}]:\n{text}"
            )

        context_str = "\n\n".join(context_parts)
        pages_list = sorted(valid_pages)
        first_page = pages_list[0] if pages_list else 1

        prompt = f"""You are a senior Class 12 board examination creator.

Generate EXACTLY 5 high-quality, conceptual, exam-style multiple-choice questions for the topic "{topic}", based ONLY on the textbook excerpts below.

TEXTBOOK EXCERPTS:
{context_str}

STRICT EXAM RULES:
1. Questions must be authentic, conceptual, and exam-oriented.
2. Never mention page numbers in questions or options.
3. Every question must have exactly four distinct options: A, B, C, D.
4. Exactly one option must be correct.
5. Explain clearly why the correct option is correct.
6. Provide a one-sentence explanation for each incorrect option.
7. Use source page {first_page} as the initial page value. The application will validate page references.

Return ONLY a valid JSON object with this structure:
{{
  "mcqs": [
    {{
      "id": 1,
      "question": "Exam-style question text?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "B",
      "explanation": "Why B is correct based on the textbook.",
      "distractor_explanations": {{
        "A": "Why A is incorrect.",
        "C": "Why C is incorrect.",
        "D": "Why D is incorrect."
      }},
      "page": {first_page}
    }}
  ]
}}"""
        return prompt

    def clean_and_parse_json(
        self,
        raw_output: str,
        fallback_pages: List[int],
    ) -> List[Dict[str, Any]]:
        text = raw_output.strip() if isinstance(raw_output, str) else ""
        data = None

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Recover a JSON object if the model adds surrounding text.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None

        raw_list = []

        if isinstance(data, dict):
            raw_list = data.get("mcqs", data.get("questions", []))

            if not raw_list and len(data) == 1:
                value = next(iter(data.values()))
                if isinstance(value, list):
                    raw_list = value

        elif isinstance(data, list):
            raw_list = data

        if not isinstance(raw_list, list):
            raw_list = []

        sanitized = []
        seen_questions = set()
        default_page = fallback_pages[0] if fallback_pages else 1

        for idx, item in enumerate(raw_list[:5], 1):
            if not isinstance(item, dict):
                continue

            question = item.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            question = question.strip()

            # Remove common accidental page references from question text.
            question = re.sub(
                r"\(?\s*According to page\s+\d+\s*,?\s*\)?",
                "",
                question,
                flags=re.IGNORECASE,
            )
            question = re.sub(
                r"\bOn page\s+\d+\s*,?\s*",
                "",
                question,
                flags=re.IGNORECASE,
            ).strip()
            normalized_question = " ".join(question.casefold().split())
            if not normalized_question or normalized_question in seen_questions:
                continue

            options = item.get("options")
            if isinstance(options, dict):
                option_values = [options.get(key) for key in ["A", "B", "C", "D"]]
            elif isinstance(options, list) and len(options) == 4:
                option_values = options
            else:
                continue
            if any(not isinstance(value, str) or not value.strip() for value in option_values):
                continue

            formatted_opts = {
                key: option_values[pos].strip()
                for pos, key in enumerate(["A", "B", "C", "D"])
            }
            if len({value.casefold() for value in formatted_opts.values()}) != 4:
                continue

            correct_ans = item.get("correct_answer", item.get("answer"))
            if not isinstance(correct_ans, str):
                continue
            correct_ans = correct_ans.strip().upper()
            if correct_ans not in formatted_opts:
                continue

            explanation = item.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                continue
            explanation = explanation.strip()
            distractors = item.get("distractor_explanations", {})
            page = item.get("page", default_page)

            # Keep page references within the retrieved textbook pages.
            if fallback_pages and page not in fallback_pages:
                page = fallback_pages[min(idx - 1, len(fallback_pages) - 1)]

            formatted_distractors = {}
            for opt_key in ["A", "B", "C", "D"]:
                if opt_key == correct_ans:
                    continue

                dist_text = (
                    distractors.get(opt_key, "")
                    if isinstance(distractors, dict)
                    else ""
                )
                if isinstance(dist_text, str) and dist_text.strip():
                    formatted_distractors[opt_key] = dist_text.strip()

            sanitized.append(
                {
                    "id": len(sanitized) + 1,
                    "question": question,
                    "options": formatted_opts,
                    "correct_answer": correct_ans,
                    "answer": correct_ans,
                    "explanation": explanation,
                    "distractor_explanations": formatted_distractors,
                    "page": int(page) if str(page).isdigit() else default_page,
                }
            )
            seen_questions.add(normalized_question)

        return sanitized

    def generate_mcqs(
        self,
        topic: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        if not retrieved_chunks:
            raise ValueError(
                "No relevant textbook context found for this topic."
            )

        fallback_pages = [chunk.get("page", 1) for chunk in retrieved_chunks]
        prompt = self.build_mcq_prompt(topic, retrieved_chunks)

        t0 = time.perf_counter()

        try:
            if self.provider == "groq":
                if not self.groq_client:
                    raise ProviderUnavailableError(
                        "The configured AI provider is unavailable."
                    )

                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=MCQ_TEMPERATURE,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )

                raw_response = (
                    response.choices[0].message.content or ""
                )

            elif self.provider == "ollama":
                payload = {
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": MCQ_TEMPERATURE,
                        "num_thread": 8,
                        "num_predict": 1000,
                        "top_p": 0.85,
                        "repeat_penalty": 1.1,
                    },
                }

                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                response.raise_for_status()
                raw_response = response.json().get("response", "")

            else:
                raise ValueError(
                    f"Unsupported LLM_PROVIDER: {self.provider}"
                )

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                APIConnectionError, APITimeoutError) as exc:
            logger.warning("MCQ provider unavailable (%s).", type(exc).__name__)
            raise ProviderUnavailableError(
                "The configured AI provider is unavailable."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            logger.warning("Ollama MCQ request failed with status %s.", status_code)
            if status_code in {401, 403, 404, 429} or status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from exc
            raise LLMGenerationError(
                "The AI provider could not generate a response."
            ) from exc
        except APIStatusError as exc:
            logger.warning("Groq MCQ request failed with status %s.", exc.status_code)
            if exc.status_code in {401, 403, 404, 429} or exc.status_code >= 500:
                raise ProviderUnavailableError(
                    "The configured AI provider is unavailable."
                ) from exc
            raise LLMGenerationError(
                "The AI provider could not generate a response."
            ) from exc
        except (ProviderUnavailableError, LLMGenerationError):
            raise
        except Exception as exc:
            logger.error("MCQ generation failed (%s).", type(exc).__name__)
            raise LLMGenerationError(
                "The AI provider could not generate a response."
            ) from exc

        llm_time = round(time.perf_counter() - t0, 4)

        mcqs = self.clean_and_parse_json(
            raw_response,
            fallback_pages,
        )
        if len(mcqs) != 5:
            raise MCQGenerationError(
                "The AI provider could not generate five valid textbook-grounded questions. "
                "Please try another topic."
            )

        return mcqs[:5], llm_time


mcq_generator = MCQGenerator()
