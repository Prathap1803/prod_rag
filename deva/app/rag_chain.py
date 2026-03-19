from deva.app.prompts import get_prompt_for_intent
from deva.app.retriever import get_retriever, format_docs
from deva.app.query_layer import build_query_layer_runnable
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(vectorstore, llm):
    retriever = get_retriever(vectorstore)
    query_layer = build_query_layer_runnable()

    # --- named functions instead of inline lambdas ---

    def get_enhanced(x):
        return x["enhanced_question"]

    def get_intent(x):
        return x.get("intent", "qa")

    def get_raw(x):
        return x.get("raw_question", "")

    def build_context(x):
        return {
            "context": format_docs(x["docs"]),
            "question": x["question"],
            "docs": x["docs"],
            "intent": x["intent"],
            "raw_question": x["raw_question"],
        }

    def generate_answer(x):
        prompt = get_prompt_for_intent(x["intent"])
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": x["question"],
            "context": x["context"],
        })

    def get_docs(x):
        return x["docs"]

    def get_intent_out(x):
        return x["intent"]

    def get_enhanced_out(x):
        return x["question"]   # at this point "question" = enhanced_question

    chain = (
        query_layer
        | RunnableParallel(
            docs=RunnableLambda(get_enhanced) | retriever,
            question=RunnableLambda(get_enhanced),
            intent=RunnableLambda(get_intent),
            raw_question=RunnableLambda(get_raw),
        )
        | RunnableLambda(build_context)
        | RunnableParallel(
            answer=RunnableLambda(generate_answer),
            sources=RunnableLambda(get_docs),
            intent=RunnableLambda(get_intent_out),
            enhanced_question=RunnableLambda(get_enhanced_out),
        )
    )

    return chain
