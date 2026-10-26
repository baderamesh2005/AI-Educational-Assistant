import os
import shutil
import time
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from backend.config import (
    ACTIVE_MODEL_NAME,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    OLLAMA_MODEL_NAME,
    UPLOADS_DIR,
    TOP_K,
)
from backend.document_processor import DocumentProcessor
from backend.embedding import embedding_manager
from backend.vector_store import vector_store
from backend.retriever import retriever
from backend.llm import LLMGenerationError, ProviderUnavailableError, llm_client
from backend.mcq import MCQGenerationError, mcq_generator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Educational Assistant API",
    description="Textbook-grounded RAG learning assistant with Ollama or Groq and FAISS",
    version="2.0.0"
)

DEFAULT_FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def get_allowed_origins(raw_origins: Optional[str] = None) -> List[str]:
    """Parses and normalizes comma-separated frontend origins, stripping whitespace and trailing slashes."""
    if raw_origins is None:
        raw_origins = os.getenv("FRONTEND_ORIGINS", "")
    if not raw_origins.strip():
        return list(DEFAULT_FRONTEND_ORIGINS)
    parsed = [
        origin.strip().rstrip("/")
        for origin in raw_origins.split(",")
        if origin.strip().rstrip("/")
    ]
    return parsed or list(DEFAULT_FRONTEND_ORIGINS)


allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
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
        content={
            "message": "AI Educational Assistant API is running",
            "provider": LLM_PROVIDER,
            "model": ACTIVE_MODEL_NAME,
        },
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.get("/health")
def health_check():
    llm_ok, llm_msg = llm_client.check_health()
    active_doc = vector_store.get_current_document()
    provider_model = {
        "ollama": OLLAMA_MODEL_NAME,
        "groq": GROQ_MODEL_NAME,
    }.get(LLM_PROVIDER, ACTIVE_MODEL_NAME)
    llm_status = {
        "provider": LLM_PROVIDER,
        "online": llm_ok,
        "message": llm_msg,
        "model": provider_model,
    }
    return JSONResponse(
        content={
            "status": "healthy" if llm_ok else "warning",
            "llm": llm_status,
            "provider": LLM_PROVIDER,
            "model": provider_model,
            # Retain the former field for clients that still read it.
            "ollama": {
                "online": llm_ok if LLM_PROVIDER == "ollama" else False,
                "message": (
                    llm_msg
                    if LLM_PROVIDER == "ollama"
                    else "Ollama is not the configured provider."
                ),
                "model": OLLAMA_MODEL_NAME,
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
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files are supported."
        )

    safe_filename = Path(file.filename).name
    # Collision-resistant storage filename
    unique_storage_filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
    saved_path = UPLOADS_DIR / unique_storage_filename

    # Read first chunk to validate PDF magic signature
    first_chunk = await file.read(1024)
    if not first_chunk.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF content. The uploaded file is not a valid PDF document."
        )

    # Stream upload in chunks while enforcing size limit
    total_size = len(first_chunk)
    try:
        with open(saved_path, "wb") as buffer:
            buffer.write(first_chunk)
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB}MB."
                    )
                buffer.write(chunk)
    except HTTPException:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise
    except Exception as io_err:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        logger.error("I/O error while saving uploaded file: %s", io_err)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while saving the uploaded file."
        )

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
        logger.warning("PDF processing rejected: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Failed to process uploaded PDF (%s)", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="Failed to process educational PDF. Please verify the document format and try again."
        )

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
    except ProviderUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="The configured AI provider is unavailable. Please check the provider configuration and try again.",
        )
    except LLMGenerationError:
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider could not generate a response. Please try again.",
        )
    except Exception as e:
        logger.error("Q&A generation failed (%s).", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the answer.",
        )

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
        try:
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
                token_payload = {
                    "type": "token",
                    "token": "I could not find sufficient information about this topic in the uploaded textbook.",
                }
                yield f"data: {json.dumps(token_payload)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'status': 'completed', 'total_sec': round(time.perf_counter() - t_start, 4)})}\n\n"
                return

            t_llm_start = time.perf_counter()
            for token in llm_client.generate_answer_stream(q, chunks):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            done_payload = {
                "type": "done",
                "status": "completed",
                "timing": {
                    "embedding_sec": metrics.get("embedding_sec", 0),
                    "retrieval_sec": metrics.get("retrieval_sec", 0),
                    "llm_sec": round(time.perf_counter() - t_llm_start, 4),
                    "total_sec": round(time.perf_counter() - t_start, 4)
                }
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except ProviderUnavailableError:
            error = {
                "type": "error",
                "message": "The configured AI provider is unavailable. Please check the provider configuration and try again.",
            }
            yield f"data: {json.dumps(error)}\n\n"
        except LLMGenerationError:
            error = {
                "type": "error",
                "message": "The configured AI provider could not generate a response. Please try again.",
            }
            yield f"data: {json.dumps(error)}\n\n"
        except Exception as e:
            logger.error("Answer stream failed (%s).", type(e).__name__)
            error = {
                "type": "error",
                "message": "An unexpected error occurred while generating the answer.",
            }
            yield f"data: {json.dumps(error)}\n\n"

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
    except ProviderUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="The configured AI provider is unavailable. Please check the provider configuration and try again.",
        )
    except MCQGenerationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LLMGenerationError:
        raise HTTPException(
            status_code=502,
            detail="The configured AI provider could not generate the questions. Please try again.",
        )
    except Exception as e:
        logger.error("MCQ endpoint failed (%s).", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating questions.",
        )

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
