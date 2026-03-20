import os

from dotenv import load_dotenv

load_dotenv(dotenv_path="deva/.env")

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("DEVA_LLM_PROVIDER", "ollama")
LLM_MODEL       = os.getenv("DEVA_LLM_MODEL", "dolphin-mistral")
LLM_TEMPERATURE = float(os.getenv("DEVA_LLM_TEMPERATURE", "0.3"))
OLLAMA_BASE_URL = os.getenv("DEVA_OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDINGS_PROVIDER = os.getenv("DEVA_EMBEDDINGS_PROVIDER", "huggingface")
EMBEDDINGS_MODEL    = os.getenv(
    "DEVA_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.getenv("DEVA_DATA_DIR", "./data")
CHROMA_DIR = os.getenv("DEVA_CHROMA_DIR", "./storage/chroma_db")

# ── Vector store ──────────────────────────────────────────────────────────────
VECTORSTORE_BACKEND = os.getenv("DEVA_VECTORSTORE", "chroma")

# ── Retriever ─────────────────────────────────────────────────────────────────
RETRIEVER_TYPE  = os.getenv("DEVA_RETRIEVER", "mmr")   # mmr | hybrid
RETRIEVER_TOP_K = int(os.getenv("DEVA_RETRIEVER_TOP_K", "5"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("DEVA_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("DEVA_CHUNK_OVERLAP", "200"))

# ── Reranking ─────────────────────────────────────────────────────────────────
COHERE_API_KEY      = os.getenv("COHERE_API_KEY")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
COHERE_RERANK_TOP_N = int(os.getenv("COHERE_RERANK_TOP_N", "5"))

# ── Langfuse ──────────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL   = os.getenv("DEVA_LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("DEVA_LOG_TO_FILE", "true")
LOG_FILE    = os.getenv("DEVA_LOG_FILE", "./logs/deva.log")
