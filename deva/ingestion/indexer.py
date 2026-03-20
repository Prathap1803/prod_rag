import hashlib
from deva.config import EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL, GEMINI_API_KEY
from deva.providers.embeddings import get_embeddings
from deva.providers.vectorstore.factory import get_vectorstore
from deva.logger import get_logger

logger = get_logger(__name__)


def get_or_create_vectorstore(reset: bool = False):
    embeddings = get_embeddings(
        provider=EMBEDDINGS_PROVIDER,
        model=EMBEDDINGS_MODEL,
        api_key=GEMINI_API_KEY,
    )
    return get_vectorstore(embeddings=embeddings, reset=reset)


def _chunk_id(chunk) -> str:
    source = chunk.metadata.get("source", "unknown")
    page   = chunk.metadata.get("page", "0")
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
    # include page number so same content on different pages stays unique
    return f"{source}::p{page}::{content_hash}"


def add_documents(vectorstore, chunks):
    if not chunks:
        logger.error("add_documents called with empty chunks list")
        raise ValueError("No document chunks provided to add to vectorstore.")

    # Deduplicate within the batch by ID (prevents DuplicateIDError)
    seen   = {}
    for chunk in chunks:
        cid = _chunk_id(chunk)
        if cid not in seen:
            seen[cid] = chunk

    unique_chunks = list(seen.values())
    unique_ids    = list(seen.keys())

    logger.info(
        f"Adding {len(unique_chunks)} unique chunks "
        f"(dropped {len(chunks) - len(unique_chunks)} duplicates)"
    )

    # Use get_or_create so existing IDs are skipped instead of erroring
    try:
        vectorstore.add_documents(unique_chunks, ids=unique_ids)
    except Exception as e:
        if "DuplicateIDError" in type(e).__name__ or "duplicate" in str(e).lower():
            # IDs already exist in the store — safe to skip
            logger.warning(
                f"Some chunks already exist in vectorstore, skipping duplicates. "
                f"Run with reset=True to rebuild from scratch."
            )
        else:
            raise

    logger.info("Chunks added successfully")
