import os

DATA_DIR = os.getenv("DEVA_DATA_DIR", "./data")
CHROMA_DIR = os.getenv("DEVA_CHROMA_DIR", "./deva_cli/storage/chroma_db")

LLM_PROVIDER = os.getenv("DEVA_LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("DEVA_LLM_MODEL", "dolphin-mistral")

EMBEDDINGS_PROVIDER = os.getenv(
    "DEVA_EMBEDDINGS_PROVIDER", "huggingface"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
