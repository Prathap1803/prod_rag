from enum import Enum
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from deva.providers.llm import get_llm
from deva.config import LLM_PROVIDER, LLM_MODEL
from deva.logger import get_logger

logger = get_logger(__name__)


class Intent(str, Enum):
    QA        = "qa"
    SUMMARIZE = "summarize"
    METADATA  = "metadata"
    CHAT      = "chat"


_INTENT_PROMPT = PromptTemplate.from_template(
    """
Classify the user query into exactly one label:
- qa        : question about document content
- summarize : request to summarize a document or topic
- metadata  : asking about file names, pages, dates, structure
- chat      : general conversation not about documents

Return ONLY the label. No explanation.

Query: {question}
"""
)

_ENHANCE_PROMPT = PromptTemplate.from_template(
    """
Rewrite the user query for better document retrieval.
Keep the original intent. Be concise and keyword-rich.
Use any conversation context below if relevant.

Intent       : {intent}
Context hint : {context_hint}
Original     : {question}

Rewritten query (one line only):
"""
)

_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    return _llm_instance


def classify_intent(question: str) -> Intent:
    logger.debug(f"Classifying intent: {question!r}")
    try:
        out = (_INTENT_PROMPT | _get_llm()).invoke({"question": question})
        label = str(out).strip().lower()
        intent = Intent(label) if label in {i.value for i in Intent} else Intent.QA
    except Exception as e:
        logger.warning(f"Intent classification failed: {e} — defaulting to qa")
        intent = Intent.QA
    logger.info(f"Intent: {intent.value}")
    return intent


def enhance_query(question: str, intent: Intent, context_hint: str = "") -> str:
    logger.debug(f"Enhancing query | intent={intent.value}")
    try:
        out = (_ENHANCE_PROMPT | _get_llm()).invoke(
            {"intent": intent.value, "question": question, "context_hint": context_hint}
        )
        enhanced = str(out).strip()
    except Exception as e:
        logger.warning(f"Query enhancement failed: {e} — using original")
        enhanced = question
    logger.info(f"Enhanced query: {enhanced!r}")
    return enhanced


def build_query_layer_runnable() -> RunnableLambda:
    def _run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw_question  = inputs["question"]
        context_hint  = inputs.get("context_hint", "")
        intent        = classify_intent(raw_question)
        enhanced      = enhance_query(raw_question, intent, context_hint)
        return {
            "raw_question":      raw_question,
            "enhanced_question": enhanced,
            "intent":            intent.value,
            "context_hint":      context_hint,
        }
    return RunnableLambda(_run)
