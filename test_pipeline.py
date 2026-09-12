import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.document_processor import DocumentProcessor
from backend.embedding import embedding_manager
from backend.vector_store import vector_store
from backend.retriever import retriever
from backend.llm import llm_client
from backend.mcq import mcq_generator

pdf_path = BASE_DIR / "sample_docs" / "Physics_Class12_SHM.pdf"

print("\n--- 1. Document Extraction & Chunking ---")
res = DocumentProcessor.extract_and_chunk_pdf(pdf_path, "Physics_Class12_SHM.pdf")
print(f"Pages: {res['total_pages']}, Chunks: {res['total_chunks']}")

print("\n--- 2. Embedding Generation & Vector Indexing ---")
texts = [c["text"] for c in res["chunks"]]
embeddings = embedding_manager.encode_texts(texts)
doc_info = vector_store.replace_with_new_document(
    res["chunks"], embeddings, "Physics_Class12_SHM.pdf", res["total_pages"]
)
print(f"Active Document Info: {doc_info}")

print("\n--- 3. Semantic Retrieval Test ---")
chunks, timing = retriever.retrieve("What is simple harmonic motion?", top_k=3)
print(f"Retrieved {len(chunks)} chunks. Latency: {timing}")
for c in chunks:
    print(f"  [Page {c['page']}] Score: {c.get('composite_score', 0):.3f} | {c['text'][:90]}...")

print("\n--- 4. Grounded Q&A Generation (Ollama llama3.2:3b) ---")
t0 = time.time()
ans, llm_sec = llm_client.generate_answer_sync("What is simple harmonic motion?", chunks)
print(f"Ollama Q&A completed in {llm_sec:.2f}s:")
print(ans)

print("\n--- 5. Out-of-Context Question Grounding Refusal Test ---")
chunks_unrelated, timing_un = retriever.retrieve("What is the capital city of France?", top_k=2)
ans_refusal, ref_sec = llm_client.generate_answer_sync("What is the capital city of France?", chunks_unrelated)
print(f"Refusal response (took {ref_sec:.2f}s):\n{ans_refusal}")

print("\n--- 6. MCQ Generation Test (5 MCQs) ---")
mcqs, mcq_llm_sec = mcq_generator.generate_mcqs("Simple Harmonic Motion", chunks)
print(f"Generated {len(mcqs)} MCQs in {mcq_llm_sec:.2f}s:")
for m in mcqs:
    print(f"\nQ{m['id']}. {m['question']} (Page {m['page']})")
    for opt_k, opt_v in m["options"].items():
        print(f"   {opt_k}. {opt_v}")
    print(f"   Correct Answer: {m['answer']}")
    print(f"   Explanation: {m['explanation']}")

print("\nALL BACKEND TESTS COMPLETED SUCCESSFULLY!")
