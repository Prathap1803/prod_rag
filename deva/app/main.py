import os
import time

from deva.config import (
    DATA_DIR,
    CHROMA_DIR,
    LLM_PROVIDER,
    LLM_MODEL,
    GEMINI_API_KEY,
)
from deva.providers.llm import get_llm
from deva.ingestion.loaders import load_documents
from deva.ingestion.splitter import split_documents
from deva.ingestion.indexer import get_or_create_vectorstore, add_documents

from deva.app.rag_chain import create_rag_chain


def main():
    # Load or create vector store
    if not os.path.exists(CHROMA_DIR):
        print("🔹 Creating vector store...")

        documents = load_documents(DATA_DIR)
        chunks = split_documents(documents)
        vectorstore = get_or_create_vectorstore(chunks)

        print(f"✅ Indexed {len(chunks)} chunks")
    else:
        print("🔹 Loading existing vector store...")
        vectorstore = get_or_create_vectorstore()

    # Create LLM
    llm = get_llm(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        #api_key=GEMINI_API_KEY,
    )
    print(llm)

    # Build RAG chain
    chain = create_rag_chain(vectorstore, llm)

    # Interactive loop
    while True:
        query = input("\nEnter your query (exit/quit to stop): ")

        if query.lower() in {"exit", "quit"}:
            print("👋 Bye!")
            break

        if len(query) > 1000:
            print("❌ Query too long")
            continue

        start = time.time()
        result = chain.invoke({"question": query})
        latency = time.time() - start

        answer = result["answer"]
        sources = result["sources"]

        print(f"\nAnswer:\n{answer}")
        print(f"\n⏱️ Latency: {latency:.2f}s")

        print("\n📚 Sources:")
        seen = set()
        for doc in sources:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = f"{src} (page {page})" if page else src

            if label not in seen:
                print(f"- {label}")
                seen.add(label)
