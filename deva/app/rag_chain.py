
from deva.app.prompts import RAG_PROMPT
from deva.app.retriever import get_retriever
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(vectorstore, llm):
    retriever = get_retriever(vectorstore)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel(
            docs=RunnableLambda(lambda x: x["question"]) | retriever,
            question=RunnableLambda(lambda x: x["question"]),
        )
        | RunnableLambda(
            lambda x: {
                "context": format_docs(x["docs"]),
                "question": x["question"],
                "docs": x["docs"],
            }
        )
        | RunnableParallel(
            answer=RAG_PROMPT | llm | StrOutputParser(),
            sources=RunnableLambda(lambda x: x["docs"]),
        )
)


    return chain
