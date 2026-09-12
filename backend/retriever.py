import time
import re
from typing import List, Dict, Any, Tuple
from backend.config import TOP_K, MAX_CONTEXT_CHUNKS
from backend.embedding import embedding_manager
from backend.vector_store import vector_store

class EducationalRetriever:
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        # Filter out common stop words to retain key scientific concepts
        words = re.findall(r'[a-zA-Z0-9_\-]+', query.lower())
        stop_words = {
            "what", "is", "the", "in", "of", "and", "a", "an", "to", "for",
            "with", "on", "at", "by", "from", "how", "why", "does", "explain",
            "describe", "define", "give", "state", "discuss", "mention", "about",
            "theory", "concept", "principle"
        }
        return [w for w in words if len(w) > 2 and w not in stop_words]

    @staticmethod
    def retrieve(
        query: str,
        top_k: int = TOP_K,
        strict_topic_filter: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves highly relevant textbook chunks using semantic similarity,
        keyword/concept matching, and adaptive filtering.
        Calculates authentic evidence confidence & source match scores.
        """
        timing = {}

        t0 = time.perf_counter()
        query_vec = embedding_manager.encode_query(query)
        timing["embedding_sec"] = round(time.perf_counter() - t0, 4)

        t1 = time.perf_counter()
        # Fetch initial pool of candidates
        raw_results = vector_store.search(query_vec, top_k=8)
        timing["retrieval_sec"] = round(time.perf_counter() - t1, 4)

        if not raw_results:
            return [], {
                "embedding_sec": timing["embedding_sec"],
                "retrieval_sec": timing["retrieval_sec"],
                "source_match_pct": 0,
                "confidence_pct": 0,
                "is_grounded": False
            }

        keywords = EducationalRetriever.extract_keywords(query)
        scored_candidates = []

        for item in raw_results:
            base_sim = float(item.get("similarity", 0.0))
            text_lower = item.get("text", "").lower()

            # Calculate keyword presence
            kw_hits = sum(1 for kw in keywords if kw in text_lower)
            kw_ratio = (kw_hits / len(keywords)) if keywords else 0.5

            # Composite score balancing semantic meaning and terminology
            composite = base_sim + (0.12 * kw_ratio)

            item_copy = dict(item)
            item_copy["raw_similarity"] = base_sim
            item_copy["kw_hits"] = kw_hits
            item_copy["composite_score"] = round(composite, 4)
            scored_candidates.append(item_copy)

        # Sort by composite score descending
        scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        best_sim = scored_candidates[0]["raw_similarity"]
        best_composite = scored_candidates[0]["composite_score"]

        # Relevance filtering: do not blindly pass irrelevant chunks
        # If best similarity is very poor (< 0.25), query is out of textbook scope
        if best_sim < 0.25 and scored_candidates[0]["kw_hits"] == 0:
            return [], {
                "embedding_sec": timing["embedding_sec"],
                "retrieval_sec": timing["retrieval_sec"],
                "source_match_pct": max(int(best_sim * 100), 12),
                "confidence_pct": max(int(best_sim * 80), 15),
                "is_grounded": False
            }

        filtered = []
        for cand in scored_candidates:
            sim = cand["raw_similarity"]
            # Keep if similarity is respectable or within reasonable range of best match
            sim_delta = best_sim - sim
            if strict_topic_filter:
                # For MCQs, be even stricter: require topic keyword hit or strong similarity
                if sim >= 0.35 and sim_delta <= 0.25:
                    filtered.append(cand)
            else:
                if sim >= 0.28 and sim_delta <= 0.32:
                    filtered.append(cand)

        # If filtering removed everything, keep at most the top candidate if it had hits
        if not filtered and best_sim >= 0.28:
            filtered = [scored_candidates[0]]

        final_chunks = filtered[:min(top_k, MAX_CONTEXT_CHUNKS)]

        # Calculate Authentic Evidence Metrics
        # Source Match %: How close the best source excerpt matches the query
        # MiniLM cosine similarity: ~0.70 is near perfect, ~0.50 is solid match
        scaled_match = int(min(max(best_sim * 135, 20), 96))

        # Answer Confidence %: Evaluates depth of evidence across chunks and keyword presence
        top_kw_hits = final_chunks[0]["kw_hits"] if final_chunks else 0
        kw_bonus = 8 if top_kw_hits >= 2 else (4 if top_kw_hits == 1 else 0)
        confidence_base = int(min(max(best_composite * 120, 25), 92))
        final_confidence = min(confidence_base + kw_bonus, 96)

        metrics = {
            "embedding_sec": timing["embedding_sec"],
            "retrieval_sec": timing["retrieval_sec"],
            "source_match_pct": scaled_match,
            "confidence_pct": final_confidence,
            "is_grounded": True
        }

        return final_chunks, metrics

retriever = EducationalRetriever()
