import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
APP_STORAGE_DIR = Path(os.getenv("APP_STORAGE_DIR", str(BASE_DIR))).expanduser()
DATA_DIR = APP_STORAGE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTOR_STORE_DIR = APP_STORAGE_DIR / "vector_store"

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


# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama settings for local development
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:3b")

# Groq settings for cloud deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

# Keep MODEL_NAME for existing imports while exposing the selected model clearly.
ACTIVE_MODEL_NAME = {
    "ollama": OLLAMA_MODEL_NAME,
    "groq": GROQ_MODEL_NAME,
}.get(LLM_PROVIDER, "unknown")
MODEL_NAME = os.getenv("MODEL_NAME", ACTIVE_MODEL_NAME)

TEMPERATURE = 0.2
MCQ_TEMPERATURE = 0.1
OLLAMA_TIMEOUT = 180  # seconds
