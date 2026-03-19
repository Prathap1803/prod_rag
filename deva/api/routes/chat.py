import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langfuse import propagate_attributes

from deva.api.schemas import ChatRequest, ChatResponse, SourceDoc
from deva.app.observability import get_langfuse_handler
from deva.app.rag_chain import create_rag_chain
from deva.ingestion.indexer import get_or_create_vectorstore
from deva.providers.llm import get_llm
from deva.config import LLM_PROVIDER, LLM_MODEL

router = APIRouter()
from deva.logger import get_logger

logger = get_logger(__name__)
# Singleton chain — built once on first request
_chain = None


def get_chain():
    global _chain
    if _chain is None:
        vectorstore = get_or_create_vectorstore()
        llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
        _chain = create_rag_chain(vectorstore, llm)
    return _chain


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f"Chat request | user={req.user_id} | session={req.session_id} | question={req.question!r}")

    chain = get_chain()

    session_id = req.session_id or str(uuid4())
    user_id = req.user_id or "api-user"
    langfuse_handler = get_langfuse_handler()

    try:
        start = time.time()
        with propagate_attributes(session_id=session_id, user_id=user_id):
            result = chain.invoke(
                {
                    "question": req.question,
                    "context_hint": req.context_hint or "",
                },
                config={"callbacks": [langfuse_handler]},
            )
        latency = round(time.time() - start, 3)
        logger.info(f"Chat completed | intent={result.get('intent')} | latency={latency}s")
    except Exception as e:
        logger.error(f"RAG chain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG chain error: {str(e)}")

    # Build sources list
    seen = set()
    sources = []
    for doc in result.get("sources", []):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{src}:{page}" if page else src
        if label not in seen:
            sources.append(SourceDoc(source=src, page=page))
            seen.add(label)
    logger.debug(f"Sources returned: {[s.source for s in sources]}")
    return ChatResponse(
        answer=result["answer"],
        intent=result.get("intent", "qa"),
        enhanced_question=result.get("enhanced_question", req.question),
        sources=sources,
        latency_seconds=latency,
    )
