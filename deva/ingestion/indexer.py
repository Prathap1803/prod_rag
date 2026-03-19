# deva/ingestion/indexer.py

import os
import shutil
from langchain_chroma import Chroma
from deva.config import EMBEDDINGS_PROVIDER, GEMINI_API_KEY, CHROMA_DIR
from deva.providers.embeddings import get_embeddings

embeddings = get_embeddings(
    provider=EMBEDDINGS_PROVIDER,
    api_key=GEMINI_API_KEY,
)


def get_or_create_vectorstore(reset: bool = False):
    """
    Load existing Chroma vectorstore or create a new one.
    
    Args:
        reset (bool): If True, deletes existing DB and creates new.
        
    Returns:
        Chroma: Vectorstore object
    """
    # Delete DB if reset
    if reset and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    
    # Make sure directory exists
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    # Load or create vectorstore
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore


def add_documents(vectorstore, chunks):
    """
    Add a list of document chunks to an existing vectorstore.
    
    Args:
        vectorstore (Chroma): The vectorstore instance
        chunks (List[Document]): List of LangChain Document objects
    """
    if not chunks:
        raise ValueError("No document chunks provided to add to vectorstore.")
    
    vectorstore.add_documents(chunks)
