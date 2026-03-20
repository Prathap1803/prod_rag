import shutil
from langchain_chroma import Chroma
from deva.config import CHROMA_DIR, VECTORSTORE_BACKEND
from deva.logger import get_logger

logger = get_logger(__name__)


def get_vectorstore(embeddings, reset: bool = False):
    backend = VECTORSTORE_BACKEND.lower()

    if backend == "chroma":
        if reset:
            import os
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
                logger.info(f"Cleared existing Chroma DB at {CHROMA_DIR}")

        logger.info(f"Initializing Chroma DB at {CHROMA_DIR}")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    raise ValueError(f"Unsupported vectorstore backend: {backend}")
