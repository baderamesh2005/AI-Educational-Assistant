import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Ensure all directories exist
for directory in [UPLOADS_DIR, EXTRACTED_DIR, CHUNKS_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Vector Store File Paths (Single Active Document Model)
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "current.index"
METADATA_PATH = VECTOR_STORE_DIR / "current_metadata.json"
CURRENT_DOC_PATH = VECTOR_STORE_DIR / "current_document.json"

# RAG & Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
MAX_CONTEXT_CHUNKS = 5

# LLM Settings (Ollama - llama3.2:3b)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
TEMPERATURE = 0.2
MCQ_TEMPERATURE = 0.1
OLLAMA_TIMEOUT = 180  # seconds
