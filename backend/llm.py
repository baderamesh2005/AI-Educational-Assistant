import json
import time
from typing import List, Dict, Any, Generator, Tuple
import requests

from backend.config import OLLAMA_BASE_URL, MODEL_NAME, TEMPERATURE, OLLAMA_TIMEOUT

class OllamaClient:
    def __init__(self):
        self.generate_url = f"{OLLAMA_BASE_URL}/api/generate"
        self.tags_url = f"{OLLAMA_BASE_URL}/api/tags"

    def check_health(self) -> Tuple[bool, str]:
        try:
            res = requests.get(self.tags_url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                base_target = MODEL_NAME.split(":")[0]
                matched = any(base_target in m for m in models)
                if matched:
                    return True, f"Ollama is online with model '{MODEL_NAME}' available."
                return True, f"Ollama is online, but model '{MODEL_NAME}' was not found in: {models}"
            return False, f"Ollama returned HTTP status {res.status_code}"
        except Exception as e:
            return False, f"Local AI model is not running. Please start Ollama. (Error: {e})"

    def build_qa_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Constructs an exam-oriented prompt tailored to student question intent.
        Strictly forbids embedding page numbers into the prose.
        """
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get("text", "").strip()
            # Retain rich context for exam answers
            if len(text) > 700:
                text = text[:700] + "..."
            context_parts.append(f"[Textbook Excerpt {idx}]:\n{text}")

        context_str = "\n\n".join(context_parts)

        prompt = f"""You are an expert AI Educational Assistant and teacher preparing Class 12 students for board examinations.
Your task is to provide an accurate, high-scoring, exam-ready answer based SOLELY on the textbook excerpts below.

CRITICAL INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the provided textbook excerpts. Do not use outside facts.
2. If the textbook excerpts do not contain enough facts to answer, respond with EXACTLY:
   "I could not find sufficient information about this topic in the uploaded textbook."
3. DO NOT MENTION PAGE NUMBERS ANYWHERE IN YOUR ANSWER. Do not say "According to page X", "On page X", or "Refer to page X". The page citation is displayed separately by the UI.
4. TAILOR THE ANSWER STRUCTURE TO THE QUESTION INTENT:
   - If the student asks to "Define X": Give a concise, standard textbook definition with units/symbols.
   - If the student asks "What is X?": Provide a clear conceptual explanation with fundamental principles.
   - If the student asks to "Explain X": Provide a thorough, descriptive, exam-ready explanation with underlying principles, key conditions, and points.
   - If the student asks to "Describe X": Give a well-structured breakdown using headings or bullet points.
   - If the student asks to "Differentiate between X and Y": Provide a clear point-by-point comparison.
   - If the student asks to "Explain the process / mechanism of X": Provide a clear step-by-step sequence.
   - For numerical questions: Use the standard structure:
     Given:
     Formula:
     Substitution:
     Calculation:
     Final Answer:
5. Maintain an academic, lucid, and encouraging tone suitable for writing in an examination.

TEXTBOOK EXCERPTS:
{context_str}

STUDENT QUESTION:
{question}

EXAM-READY ANSWER:"""
        return prompt

    def generate_answer_sync(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        prompt = self.build_qa_prompt(question, retrieved_chunks)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": TEMPERATURE,
                "num_thread": 8,
                "num_predict": 550,
                "top_p": 0.9,
            }
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Local AI model is not running. Please start Ollama.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama generation timed out after {OLLAMA_TIMEOUT} seconds.")
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")

        llm_time = round(time.perf_counter() - t0, 4)
        return answer, llm_time

    def generate_answer_stream(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        prompt = self.build_qa_prompt(question, retrieved_chunks)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "temperature": TEMPERATURE,
                "num_thread": 8,
                "num_predict": 550,
                "top_p": 0.9,
            }
        }

        try:
            with requests.post(self.generate_url, json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk_json = json.loads(line)
                        token = chunk_json.get("response", "")
                        yield token
                        if chunk_json.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield "\n\n**Error:** Local AI model is not running. Please start Ollama."
        except requests.exceptions.Timeout:
            yield f"\n\n**Error:** Ollama generation timed out after {OLLAMA_TIMEOUT} seconds."
        except Exception as e:
            yield f"\n\n**Error during generation:** {str(e)}"

llm_client = OllamaClient()
