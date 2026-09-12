import json
import re
import time
from typing import List, Dict, Any, Tuple
import requests

from backend.config import OLLAMA_BASE_URL, MODEL_NAME, MCQ_TEMPERATURE, OLLAMA_TIMEOUT

class MCQGenerator:
    def __init__(self):
        self.generate_url = f"{OLLAMA_BASE_URL}/api/generate"

    def build_mcq_prompt(self, topic: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        valid_pages = set()
        for idx, chunk in enumerate(retrieved_chunks, 1):
            page = chunk.get("page", 1)
            valid_pages.add(page)
            text = chunk.get("text", "").strip()
            if len(text) > 500:
                text = text[:500] + "..."
            context_parts.append(f"[Textbook Content {idx}]:\n{text}")

        context_str = "\n\n".join(context_parts)
        pages_list = sorted(list(valid_pages))
        first_page = pages_list[0] if pages_list else 1

        prompt = f"""You are a senior Class 12 board examination creator.
Generate EXACTLY 5 high-quality, conceptual, exam-style Multiple Choice Questions (MCQs) for the topic "{topic}" based ONLY on these excerpts:

{context_str}

STRICT EXAM RULES:
1. Every question must be an authentic exam question (e.g. "Who proposed...", "Which of the following is correct...", "What is the primary condition for...", "Which statement accurately describes...").
2. NEVER MENTION PAGE NUMBERS IN ANY QUESTION OR OPTION. (DO NOT write "According to page X" or "What is on page X").
3. Each question must have EXACTLY 4 distinct options (A, B, C, D).
4. Exactly ONE option must be correct.
5. Provide a clear, thorough "explanation" for why the correct option is correct.
6. Provide "distractor_explanations" with a clear 1-sentence explanation of why EACH incorrect option is incorrect.
7. Set "page" to the source textbook page (e.g. {first_page}).

Respond with ONLY this JSON object:
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

    def clean_and_parse_json(self, raw_output: str, fallback_pages: List[int]) -> List[Dict[str, Any]]:
        text = raw_output.strip()
        data = None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    pass

        raw_list = []
        if isinstance(data, dict):
            raw_list = data.get("mcqs", data.get("questions", []))
            if not raw_list and len(data.keys()) == 1:
                val = list(data.values())[0]
                if isinstance(val, list):
                    raw_list = val
        elif isinstance(data, list):
            raw_list = data

        sanitized = []
        default_page = fallback_pages[0] if fallback_pages else 1

        for idx, item in enumerate(raw_list[:5], 1):
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            # Clean accidental page number mentions from question text
            question = re.sub(r'\(?According to page \d+\,?\)?\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'On page \d+\,?\s*', '', question, flags=re.IGNORECASE)

            options = item.get("options", {})
            correct_ans = str(item.get("correct_answer", item.get("answer", "A"))).strip().upper()
            explanation = str(item.get("explanation", "")).strip()
            distractors = item.get("distractor_explanations", {})
            page = item.get("page", default_page)

            if fallback_pages and page not in fallback_pages:
                page = fallback_pages[min(idx - 1, len(fallback_pages) - 1)]

            formatted_opts = {}
            if isinstance(options, dict):
                for key in ["A", "B", "C", "D"]:
                    formatted_opts[key] = str(options.get(key, f"Option {key}")).strip()
            elif isinstance(options, list) and len(options) >= 4:
                for o_idx, key in enumerate(["A", "B", "C", "D"]):
                    formatted_opts[key] = str(options[o_idx]).strip()
            else:
                formatted_opts = {
                    "A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"
                }

            if correct_ans not in ["A", "B", "C", "D"]:
                correct_ans = "A"

            # Ensure every wrong option has an explanation
            formatted_distractors = {}
            for opt_key in ["A", "B", "C", "D"]:
                if opt_key != correct_ans:
                    dist_text = distractors.get(opt_key, "") if isinstance(distractors, dict) else ""
                    if not dist_text:
                        dist_text = f"Option {opt_key} does not accurately represent the principles described in the textbook for this topic."
                    formatted_distractors[opt_key] = dist_text

            if question:
                sanitized.append({
                    "id": idx,
                    "question": question,
                    "options": formatted_opts,
                    "correct_answer": correct_ans,
                    "answer": correct_ans,  # Backward compatibility
                    "explanation": explanation or f"Option {correct_ans} is the correct principle stated in the textbook.",
                    "distractor_explanations": formatted_distractors,
                    "page": int(page) if str(page).isdigit() else default_page
                })

        return sanitized

    def generate_mcqs(self, topic: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        if not retrieved_chunks:
            raise ValueError("No relevant textbook context found for this topic.")

        fallback_pages = [c.get("page", 1) for c in retrieved_chunks]
        prompt = self.build_mcq_prompt(topic, retrieved_chunks)

        payload = {
    "model": MODEL_NAME,
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
    }
}

        t0 = time.perf_counter()
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            raw_response = data.get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Local AI model is not running. Please start Ollama.")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama MCQ generation timed out after {OLLAMA_TIMEOUT} seconds.")
        except Exception as e:
            raise RuntimeError(f"Ollama MCQ generation failed: {e}")

        llm_time = round(time.perf_counter() - t0, 4)
        mcqs = self.clean_and_parse_json(raw_response, fallback_pages)

        # Fallback safeguard: ensure exactly 5 questions
        if len(mcqs) < 5:
            existing_count = len(mcqs)
            for i in range(existing_count + 1, 6):
                chunk_ref = retrieved_chunks[(i - 1) % len(retrieved_chunks)]
                c_text = chunk_ref.get("text", "")
                c_page = chunk_ref.get("page", 1)
                first_sent = c_text.split(".")[0] if "." in c_text else c_text[:80]
                mcqs.append({
                    "id": i,
                    "question": f"Which of the following statements is true regarding {topic}?",
                    "options": {
                        "A": first_sent.strip() or f"Key definition for {topic}",
                        "B": f"An opposite statement not supported for {topic}",
                        "C": "Energy is destroyed without conservation",
                        "D": "The system parameters remain totally static"
                    },
                    "correct_answer": "A",
                    "answer": "A",
                    "explanation": f"According to the textbook: '{first_sent.strip()}'.",
                    "distractor_explanations": {
                        "B": f"This statement contradicts the principles of {topic}.",
                        "C": "Conservation laws dictate that energy cannot be destroyed.",
                        "D": "The system undergoes dynamic periodic changes rather than remaining static."
                    },
                    "page": c_page
                })

        return mcqs[:5], llm_time

mcq_generator = MCQGenerator()
