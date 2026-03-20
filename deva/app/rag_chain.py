# ✅ Full updated rag_chain.py
from deva.app.prompts import get_prompt_for_intent
from deva.app.retriever import get_retriever, format_docs
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from deva.logger import get_logger

logger = get_logger(__name__)


def create_rag_chain(vectorstore, llm):
    retriever = get_retriever(vectorstore)

    def _get_search_query(x: dict) -> str:
        # Safe fallback: use enhanced_question if present, else raw question
        q = x.get("enhanced_question") or x.get("question", "")
        logger.debug(f"RAG chain search query: {q!r}")
        return q

    def _get_intent(x: dict) -> str:
        return x.get("intent", "qa")

    def _build_context(x: dict) -> dict:
        return {
            "context":  format_docs(x["docs"]),
            "question": x["question"],
            "docs":     x["docs"],
            "intent":   x["intent"],
            "enhanced_question":  x.get("enhanced_question", x["question"]),
        } 

    def _generate_answer(x: dict) -> str:
        prompt = get_prompt_for_intent(x["intent"])
        return (prompt | llm | StrOutputParser()).invoke(x)

    chain = (
        RunnableParallel(
            docs=RunnableLambda(_get_search_query) | retriever,
            question=RunnableLambda(_get_search_query),
            intent=RunnableLambda(_get_intent),
             enhanced_question=RunnableLambda(          
                lambda x: x.get("enhanced_question") or x.get("question", "")
            ),
        )
        | RunnableLambda(_build_context)
        | RunnableParallel(
            answer=RunnableLambda(_generate_answer),
            
            sources=RunnableLambda(lambda x: x["docs"]),
            intent=RunnableLambda(lambda x: x["intent"]),
        )
    )

    return chain