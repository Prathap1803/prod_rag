from typing import Any

from deva.config import RETRIEVER_TYPE


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] ({source}) {doc.page_content}")
    return "\n\n".join(formatted)


def get_vector_retriever(vectorstore: Any):
    """
    Default vector-only retriever using MMR over the given vectorstore.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "lambda_mult": 0.7, "score_threshold": 0.75,},
    )


def get_hybrid_retriever(vectorstore: Any):
    """
    Placeholder for a hybrid (dense + lexical) retriever.

    For now this just delegates to the vector retriever. You can extend this
    to:
      - query a BM25/FTS index alongside the vectorstore
      - merge scores
      - pass merged docs through a reranker model
    """
    # TODO: implement BM25 + rerank and return a custom retriever object
    return get_vector_retriever(vectorstore)


def get_retriever(vectorstore: Any):
    """
    Factory for retrievers based on DEVA_RETRIEVER env var.

    - "mmr"   -> vector-only MMR retriever (default)
    - "hybrid" -> hybrid retriever (stubbed to vector-only for now)
    """
    if RETRIEVER_TYPE == "hybrid":
        return get_hybrid_retriever(vectorstore)
    # Fallback: vector-only
    return get_vector_retriever(vectorstore)
