from dotenv import load_dotenv, find_dotenv
load_dotenv(("deva/.env"))
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DEVA_DATA_DIR", "deva/data")
CHROMA_DIR = os.getenv("DEVA_CHROMA_DIR", "./deva/storage/chroma_db")

LLM_PROVIDER = os.getenv("DEVA_LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("DEVA_LLM_MODEL", "dolphin-mistral")

EMBEDDINGS_PROVIDER = os.getenv(
    "DEVA_EMBEDDINGS_PROVIDER", "huggingface"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTORSTORE_BACKEND = os.getenv("DEVA_VECTORSTORE", "chroma")


PINECONE_API_KEY = os.getenv("DEVA_PINECONE_API_KEY", None)
PINECONE_ENV = os.getenv("DEVA_PINECONE_ENV", "us-west1-gcp")  # default env
PINECONE_INDEX = os.getenv("DEVA_PINECONE_INDEX", "deva_index")

RETRIEVER_TYPE = os.getenv("DEVA_RETRIEVER", "mmr")  # mmr | hybrid
# Retriever 
RETRIEVER_TYPE = os.getenv("DEVA_RETRIEVER", "mmr")  # mmr | hybrid

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
COHERE_RERANK_TOP_N = int(os.getenv("COHERE_RERANK_TOP_N", "5"))

# Chunking
CHUNK_SIZE = int(os.getenv("DEVA_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("DEVA_CHUNK_OVERLAP", "200"))

# Logging
LOG_LEVEL = os.getenv("DEVA_LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("DEVA_LOG_TO_FILE", "true")
LOG_FILE_PATH = os.getenv("DEVA_LOG_FILE", "./logs/deva.log")
