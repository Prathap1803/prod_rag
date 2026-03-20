from typing import Any, List, Optional
from langchain_core.documents import Document
from deva.config import RETRIEVER_TYPE, RETRIEVER_TOP_K, COHERE_API_KEY, COHERE_RERANK_MODEL, COHERE_RERANK_TOP_N
from deva.logger import get_logger

logger = get_logger(__name__)

_cohere_client = None


def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        if not COHERE_API_KEY:
            raise RuntimeError(
                "COHERE_API_KEY not set. Set it or switch DEVA_RETRIEVER=mmr"
            )
        import cohere
        _cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
    return _cohere_client


def cohere_rerank(query: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []
    logger.debug(f"Reranking {len(docs)} docs with Cohere")
    client = _get_cohere_client()
    resp = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[d.page_content for d in docs],
        top_n=min(COHERE_RERANK_TOP_N, len(docs)),
    )
    reranked = [docs[r.index] for r in resp.results]
    logger.debug(f"Reranked to {len(reranked)} docs")
    return reranked


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] ({source})\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_vector_retriever(vectorstore: Any):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "lambda_mult": 0.7},
    )


def get_hybrid_retriever(vectorstore: Any):
    base = get_vector_retriever(vectorstore)

    class HybridRetriever:
        def get_relevant_documents(self, query: str) -> List[Document]:
            docs = base.get_relevant_documents(query)
            return cohere_rerank(query, docs)

        # LangChain v0.2+ async support
        async def aget_relevant_documents(self, query: str) -> List[Document]:
            return self.get_relevant_documents(query)

    return HybridRetriever()


def get_retriever(vectorstore: Any):
    logger.info(f"Retriever type: {RETRIEVER_TYPE}")
    if RETRIEVER_TYPE == "hybrid":
        return get_hybrid_retriever(vectorstore)
    return get_vector_retriever(vectorstore)
