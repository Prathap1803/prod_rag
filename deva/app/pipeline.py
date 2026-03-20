# deva/app/pipeline.py
from langchain_core.runnables import RunnableSerializable
from deva.app.query_layer import build_query_layer_runnable
from deva.app.rag_chain import create_rag_chain
from deva.logger import get_logger

logger = get_logger(__name__)


def build_full_pipeline(vectorstore, llm) -> RunnableSerializable:
    """
    Wires query layer → RAG chain into one single pipeline.

    Input  : {"question": str, "context_hint": str (optional)}
    Output : {"answer": str, "sources": list, "intent": str}
    """
    query_layer = build_query_layer_runnable()
    rag_chain   = create_rag_chain(vectorstore, llm)

    # query_layer output dict flows directly into rag_chain
    pipeline = query_layer | rag_chain

    logger.info("Full MLOps pipeline built: query_layer | rag_chain")
    return pipeline