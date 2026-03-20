import hashlib
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from deva.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _doc_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def load_documents(folder_path: str):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Data directory not found: {folder_path}")

    documents = []
    logger.info(f"Loading documents from: {folder_path}")

    for file_name in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported file: {file_name}")
            continue

        file_path = os.path.join(folder_path, file_name)
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["doc_hash"] = _doc_hash(doc.page_content)

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} page(s) from: {file_name}")

        except Exception as e:
            logger.error(f"Failed to load {file_name}: {e}", exc_info=True)

    if not documents:
        logger.error("No valid documents found in folder")
        raise ValueError(f"No valid documents found in: {folder_path}")

    logger.info(f"Total pages loaded: {len(documents)}")
    return documents
