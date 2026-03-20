from langchain_text_splitters import RecursiveCharacterTextSplitter
from deva.config import CHUNK_SIZE, CHUNK_OVERLAP
from deva.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents):
    logger.info(
        f"Splitting {len(documents)} page(s) | "
        f"chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks
