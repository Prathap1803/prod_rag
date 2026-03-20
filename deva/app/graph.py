"""
LangGraph wrapper around the RAG chain.
Provides a graph-based entry point — easy to extend with more nodes later
(e.g. guardrail node, self-correction node, human-in-the-loop).
"""
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from deva.config import LLM_PROVIDER, LLM_MODEL
from deva.providers.llm import get_llm
from deva.ingestion.indexer import get_or_create_vectorstore
from deva.app.rag_chain import create_rag_chain
from deva.logger import get_logger

logger = get_logger(__name__)


class RagState(TypedDict):
    question:          str
    context_hint:      str
    enhanced_question: Optional[str]
    intent:            Optional[str]
    answer:            Optional[str]
    sources:           Optional[list]


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        vs  = get_or_create_vectorstore()
        llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
        _chain = create_rag_chain(vs, llm)
    return _chain


def rag_node(state: RagState) -> RagState:
    logger.debug(f"rag_node: question={state['question']!r}")
    result = _get_chain().invoke({
        "question":     state["question"],
        "context_hint": state.get("context_hint", ""),
    })
    return {
        **state,
        "enhanced_question": result.get("enhanced_question"),
        "intent":            result.get("intent", "qa"),
        "answer":            result.get("answer"),
        "sources":           result.get("sources", []),
    }


def build_graph():
    g = StateGraph(RagState)
    g.add_node("rag", rag_node)
    g.add_edge(START, "rag")
    g.add_edge("rag", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph compiled")
    return _graph


def ask(question: str, context_hint: str = "") -> dict:
    graph = get_graph()
    state = graph.invoke({
        "question":     question,
        "context_hint": context_hint,
        "enhanced_question": None,
        "intent":       None,
        "answer":       None,
        "sources":      None,
    })
    return state
