
# 🏗️ Deva Architecture

This document explains Deva’s internal architecture and design philosophy.

---

## 🎯 Design Goals

- Local-first AI
- Modular & extensible
- Easy LLM switching
- Privacy-focused
- CLI-driven UX

---

## 📦 High-Level Flow
```
User
└── CLI (Typer)
├── Ingestion Pipeline
│ ├── Document Loaders
│ ├── Text Splitter
│ └── Vector Store (Chroma)
└── RAG Pipeline
├── Retriever
├── Prompt Template
└── LLM Provider

```

## 📁 Project Structure
```
deva/
├── cli.py # CLI entry point
├── config.py # Environment & defaults
├── providers/ # Pluggable backends
│ ├── llm.py
│ ├── embeddings.py
│ └── vectorstore.py
├── app/
│ ├── main.py # App orchestration
│ ├── rag_chain.py # RAG pipeline
│ └── retriever.py
└── ingestion/
├── loaders.py
├── splitter.py
└── indexer.py
```

---

## 📥 Ingestion Pipeline

### Document Loading
- File-type detection by extension
- Uses LangChain loaders
- Outputs `Document` objects

### Chunking
- RecursiveCharacterTextSplitter
- Chunk size: 1000
- Overlap: 200

### Vector Store
- Local Chroma database
- Persistent on disk
- Incremental or full rebuild

---

## 🧠 RAG Pipeline

1. User query
2. Retriever fetches top-k relevant chunks
3. Prompt enforces context-only answers
4. LLM generates response
5. Output returned as text

---

## 🔌 Provider Abstraction

### LLM Providers
- Ollama (local)
- Gemini (cloud)
- Easily extensible

### Embeddings Providers
- HuggingFace
- Gemini

Selected via environment variables or CLI flags.

---

## 🔒 Privacy & Security

- No telemetry
- No data leaves device by default
- API keys never committed

---

## 🔧 Extensibility

Add new providers by implementing:
- `get_llm()`
- `get_embeddings()`

Core logic remains untouched.

---

## 🚀 Future Improvements

- Streaming support
- MMR retrieval
- Multi-vector stores
- Tool calling
