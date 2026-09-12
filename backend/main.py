import os
import shutil
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from backend.config import UPLOADS_DIR, TOP_K, MODEL_NAME
from backend.document_processor import DocumentProcessor
from backend.embedding import embedding_manager
from backend.vector_store import vector_store
from backend.retriever import retriever
from backend.llm import llm_client
from backend.mcq import mcq_generator

app = FastAPI(
    title="AI Educational Assistant API",
    description="RAG-powered textbook learning assistant with Ollama and FAISS",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class QuestionRequest(BaseModel):
    question: str

class MCQRequest(BaseModel):
    topic: str

@app.get("/")
def root():
    return JSONResponse(
        content={"message": "AI Educational Assistant API is running", "model": MODEL_NAME},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/health")
def health_check():
    ollama_ok, ollama_msg = llm_client.check_health()
    active_doc = vector_store.get_current_document()
    return JSONResponse(
        content={
            "status": "healthy" if ollama_ok else "warning",
            "ollama": {
                "online": ollama_ok,
                "message": ollama_msg,
                "model": MODEL_NAME
            },
            "vector_store": {
                "has_active_doc": vector_store.has_active_document(),
                "current_document": active_doc
            }
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/current-document")
def get_current_document():
    doc = vector_store.get_current_document()
    if not doc:
        return JSONResponse(
            content={"filename": None, "pages": 0, "chunks": 0, "status": "none"},
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
    return JSONResponse(
        content=doc,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files are supported."
        )

    # Save uploaded file
    safe_filename = Path(file.filename).name
    saved_path = UPLOADS_DIR / safe_filename
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Extract and Chunk PDF
        processed = DocumentProcessor.extract_and_chunk_pdf(saved_path, safe_filename)
        chunks = processed["chunks"]
        total_pages = processed["total_pages"]

        # 2. Generate Embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_manager.encode_texts(chunk_texts)

        # 3. Replace Active Vector Store Index
        doc_info = vector_store.replace_with_new_document(
            chunks=chunks,
            embeddings=embeddings,
            filename=safe_filename,
            total_pages=total_pages
        )

        return JSONResponse(
            content={
                "success": True,
                "filename": safe_filename,
                "pages": total_pages,
                "chunks": len(chunks),
                "message": "PDF processed successfully"
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/ask")
def ask_question(req: QuestionRequest):
    if not vector_store.has_active_document():
        raise HTTPException(
            status_code=400,
            detail="Please upload a textbook PDF first."
        )

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    t_start = time.perf_counter()

    # 1. Retrieve relevant chunks & calculate evidence confidence
    chunks, metrics = retriever.retrieve(q, top_k=TOP_K)
    
    if not chunks or not metrics.get("is_grounded", False):
        return JSONResponse(
            content={
                "question": q,
                "answer": "I could not find sufficient information about this topic in the uploaded textbook.",
                "primary_page": None,
                "confidence_pct": metrics.get("confidence_pct", 15),
                "source_match_pct": metrics.get("source_match_pct", 12),
                "sources": [],
                "timing": {
                    "embedding_sec": metrics.get("embedding_sec", 0),
                    "retrieval_sec": metrics.get("retrieval_sec", 0),
                    "llm_sec": 0,
                    "total_sec": round(time.perf_counter() - t_start, 4)
                }
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    # 2. Generate Descriptive Exam Answer
    try:
        answer, llm_sec = llm_client.generate_answer_sync(q, chunks)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Local AI model is not running. Please start Ollama.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_sec = round(time.perf_counter() - t_start, 4)

    # Format sources (most relevant first)
    sources = []
    for c in chunks:
        sources.append({
            "page": c.get("page", 1),
            "chunk_id": c.get("chunk_id", 0),
            "text": c.get("text", "")[:320] + "..." if len(c.get("text", "")) > 320 else c.get("text", ""),
            "similarity": round(c.get("raw_similarity", c.get("similarity", 0.0)), 3)
        })

    primary_page = chunks[0].get("page", 1) if chunks else None

    return JSONResponse(
        content={
            "question": q,
            "answer": answer,
            "primary_page": primary_page,
            "confidence_pct": metrics.get("confidence_pct", 85),
            "source_match_pct": metrics.get("source_match_pct", 80),
            "sources": sources,
            "timing": {
                "embedding_sec": metrics.get("embedding_sec", 0),
                "retrieval_sec": metrics.get("retrieval_sec", 0),
                "llm_sec": llm_sec,
                "total_sec": total_sec
            }
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.post("/ask/stream")
def ask_question_stream(req: QuestionRequest):
    if not vector_store.has_active_document():
        raise HTTPException(
            status_code=400,
            detail="Please upload a textbook PDF first."
        )

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_stream():
        t_start = time.perf_counter()
        chunks, metrics = retriever.retrieve(q, top_k=TOP_K)
        
        sources = [
            {
                "page": c.get("page", 1),
                "chunk_id": c.get("chunk_id", 0),
                "text": c.get("text", "")[:320] + "..." if len(c.get("text", "")) > 320 else c.get("text", ""),
                "similarity": round(c.get("raw_similarity", c.get("similarity", 0.0)), 3)
            }
            for c in chunks
        ]
        primary_page = chunks[0].get("page", 1) if chunks else None

        # Emit meta event with sources and evidence confidence
        init_payload = {
            "type": "meta",
            "primary_page": primary_page,
            "confidence_pct": metrics.get("confidence_pct", 20 if not chunks else 85),
            "source_match_pct": metrics.get("source_match_pct", 15 if not chunks else 80),
            "sources": sources,
            "embedding_sec": metrics.get("embedding_sec", 0),
            "retrieval_sec": metrics.get("retrieval_sec", 0)
        }
        yield f"data: {json.dumps(init_payload)}\n\n"

        if not chunks or not metrics.get("is_grounded", False):
            yield f"data: {json.dumps({'type': 'token', 'token': 'I could not find sufficient information about this topic in the uploaded textbook.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'total_sec': round(time.perf_counter() - t_start, 4)})}\n\n"
            return

        # Stream tokens
        t_llm_start = time.perf_counter()
        for token in llm_client.generate_answer_stream(q, chunks):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        llm_sec = round(time.perf_counter() - t_llm_start, 4)
        total_sec = round(time.perf_counter() - t_start, 4)

        done_payload = {
            "type": "done",
            "timing": {
                "embedding_sec": metrics.get("embedding_sec", 0),
                "retrieval_sec": metrics.get("retrieval_sec", 0),
                "llm_sec": llm_sec,
                "total_sec": total_sec
            }
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8"
    )

@app.post("/generate-mcq")
def generate_mcqs_endpoint(req: MCQRequest):
    if not vector_store.has_active_document():
        raise HTTPException(
            status_code=400,
            detail="Please upload a textbook PDF first."
        )

    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    t_start = time.perf_counter()

    # Topic-specific filtered retrieval
    chunks, metrics = retriever.retrieve(topic, top_k=TOP_K, strict_topic_filter=True)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant textbook content found for topic: '{topic}'"
        )

    try:
        mcqs, llm_sec = mcq_generator.generate_mcqs(topic, chunks)
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Local AI model is not running. Please start Ollama.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_sec = round(time.perf_counter() - t_start, 4)

    return JSONResponse(
        content={
            "topic": topic,
            "mcqs": mcqs,
            "timing": {
                "embedding_sec": metrics.get("embedding_sec", 0),
                "retrieval_sec": metrics.get("retrieval_sec", 0),
                "llm_sec": llm_sec,
                "total_sec": total_sec
            }
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )
