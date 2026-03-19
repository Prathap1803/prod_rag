# deva/ingestion/indexer.py
from deva.config import EMBEDDINGS_PROVIDER, GEMINI_API_KEY
from deva.providers.embeddings import get_embeddings
from deva.providers.vectorstore.factory import get_vectorstore



def get_or_create_vectorstore(reset=False):
    """
    Load or create a vector store (local or remote)
    """
    return get_vectorstore(
    embeddings = get_embeddings(
    provider=EMBEDDINGS_PROVIDER,
    api_key=GEMINI_API_KEY,
    ),
        reset=reset
    )

def add_documents(vectorstore, chunks):
    """
    Add document chunks to the active vector store.

    Args:
        vectorstore: LangChain-compatible vector store
        chunks (list[Document]): List of LangChain Document objects
    """
    if not chunks:
        raise ValueError("No document chunks provided to add to vectorstore.")

    vectorstore.add_documents(chunks)

