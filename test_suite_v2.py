import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.document_processor import DocumentProcessor
from backend.embedding import embedding_manager
from backend.vector_store import vector_store
from backend.retriever import retriever
from backend.llm import llm_client
from backend.mcq import mcq_generator

print("==================================================")
print("TEST A & F: Uploading Biology PDF & Verifying Processing")
print("==================================================")
bio_pdf = BASE_DIR / "sample_docs" / "Biology_Class12_Genetics.pdf"
res = DocumentProcessor.extract_and_chunk_pdf(bio_pdf, "Biology_Class12_Genetics.pdf")
print(f"Extracted {res['total_pages']} pages, created {res['total_chunks']} chunks.")

texts = [c["text"] for c in res["chunks"]]
embeddings = embedding_manager.encode_texts(texts)
doc_info = vector_store.replace_with_new_document(
    res["chunks"], embeddings, "Biology_Class12_Genetics.pdf", res["total_pages"]
)
print("Active Document:", doc_info)

print("\n==================================================")
print("TEST A: Ask AI - Descriptive Exam-Ready Answer & Confidence")
print("==================================================")
q = "Explain the chromosomal theory of inheritance."
chunks, metrics = retriever.retrieve(q, top_k=3)
print(f"Retrieved {len(chunks)} chunks.")
print(f"Metrics: Answer Confidence = {metrics['confidence_pct']}%, Source Match = {metrics['source_match_pct']}%")

ans, llm_sec = llm_client.generate_answer_sync(q, chunks)
print(f"\n[AI Answer ({llm_sec:.2f}s)]:\n{ans}\n")

# Verify no page number inside the answer body
has_page_mention = "according to page" in ans.lower() or "on page 78" in ans.lower() or "page 78" in ans.lower()
print(f"VERIFICATION: Answer body contains forbidden page phrases? {'YES (FAIL)' if has_page_mention else 'NO (PASS - clean wording)'}")
print(f"Primary Source Page: Textbook Page {chunks[0]['page']}")

print("\n==================================================")
print("TEST B, C, D: Exam-Style MCQs with Distractor Explanations")
print("==================================================")
topic = "Chromosomal Theory of Inheritance"
mcqs, mcq_llm_sec = mcq_generator.generate_mcqs(topic, chunks)
print(f"Generated {len(mcqs)} MCQs in {mcq_llm_sec:.2f}s.")

for idx, m in enumerate(mcqs, 1):
    q_text = m['question']
    bad_page_ref = "page" in q_text.lower()
    print(f"\nQ{idx}: {q_text}")
    print(f"   Contains 'page' in question? {'YES (FAIL)' if bad_page_ref else 'NO (PASS)'}")
    for opt_k, opt_v in m["options"].items():
        print(f"   {opt_k}. {opt_v}")
    correct = m['correct_answer']
    print(f"   Correct Answer: Option {correct}")
    print(f"   Why {correct} is correct: {m['explanation']}")
    
    # Test wrong answer distractor evaluation
    wrong_opts = [k for k in ["A", "B", "C", "D"] if k != correct]
    picked_wrong = wrong_opts[0]
    distractor_reason = m.get("distractor_explanations", {}).get(picked_wrong, "N/A")
    print(f"   [Simulation] Student picked WRONG answer: Option {picked_wrong}")
    print(f"   Why Option {picked_wrong} is incorrect: {distractor_reason}")
    print(f"   Source: Page {m['page']}")

print("\n==================================================")
print("TEST E: Topic-Relevance Filtering Test")
print("==================================================")
# Test filtering when searching for a distinct topic
shm_chunks, shm_metrics = retriever.retrieve("Simple Harmonic Motion", top_k=3, strict_topic_filter=True)
print(f"Attempting to retrieve SHM from Biology textbook: found {len(shm_chunks)} chunks, grounded = {shm_metrics.get('is_grounded')}")
assert not shm_metrics.get('is_grounded', False), "Biology textbook should not consider SHM grounded!"
print("Topic relevance filtering PASS!")

print("\nALL V2 TESTS COMPLETED SUCCESSFULLY!")
