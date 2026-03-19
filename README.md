# 🧠 Deva — Local RAG API

A production-grade, local Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and FastAPI.
Chat with your own documents using local or cloud LLMs — with monitoring, evaluation, and guardrails built in.

> **Privacy first:** No data leaves your machine unless you choose an online LLM or monitoring service.

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📂 | **Document ingestion** | PDF, DOCX, TXT support with smart chunking |
| 🧠 | **Local vector database** | ChromaDB with persistent embeddings |
| 🔍 | **Hybrid retrieval** | MMR vector search + optional Cohere reranking |
| 🎯 | **Query intelligence** | Intent classification + query enhancement before retrieval |
| 🤖 | **Pluggable LLM support** | Local (Ollama) or Online (Gemini) |
| 🔁 | **LangGraph orchestration** | Graph-based pipeline with extensible nodes |
| 📡 | **FastAPI REST API** | `/chat`, `/ingest`, and `/health` endpoints |
| 📊 | **Langfuse monitoring** | Full trace observability per session and user |
| 🧪 | **RAGAS evaluation** | Faithfulness, answer relevancy, context precision & recall |
| 🛡️ | **Guardrails** | Prompt-level safety + output enforcement |
| 📝 | **Structured logging** | Levelled, rotating file + stdout logs |

---

## 🏗️ Architecture

```
POST /api/v1/chat
        │
        ▼
Query Layer (intent classify + query enhance)
        │
        ▼
Retriever (MMR vector search → optional Cohere rerank)
        │
        ▼
Intent-aware Prompt selection
        │
        ▼
LLM (Ollama / Gemini)
        │
        ▼
Guardrails (output validation)
        │
        ▼
ChatResponse + Langfuse trace
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph + LangChain LCEL |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| LLM (local) | Ollama |
| LLM (online) | Google Gemini |
| Reranking | Cohere Rerank API |
| Monitoring | Langfuse |
| Evaluation | RAGAS |
| Logging | Python logging (rotating file) |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Prathap1803/deva-cli.git
cd deva-cli
pip install -e .
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
# LLM
DEVA_LLM_PROVIDER=ollama
DEVA_LLM_MODEL=dolphin-mistral

# Paths
DEVA_DATA_DIR=./data
DEVA_CHROMA_DIR=./deva_cli/storage/chroma_db

# Embeddings
DEVA_EMBEDDINGS_PROVIDER=huggingface

# Retriever: mmr | hybrid
DEVA_RETRIEVER=mmr

# Chunking
DEVA_CHUNK_SIZE=1000
DEVA_CHUNK_OVERLAP=200

# Logging
DEVA_LOG_LEVEL=INFO
DEVA_LOG_TO_FILE=true
DEVA_LOG_FILE=./logs/deva.log

# Optional: Cohere reranking (only if DEVA_RETRIEVER=hybrid)
# COHERE_API_KEY=your_key

# Optional: Langfuse monitoring
# LANGFUSE_PUBLIC_KEY=your_key
# LANGFUSE_SECRET_KEY=your_key

# Optional: Gemini
# DEVA_LLM_PROVIDER=gemini
# DEVA_LLM_MODEL=gemini-1.5-pro
# GEMINI_API_KEY=your_key
```

### 3. Pull Ollama model (if using local LLM)

```bash
ollama pull dolphin-mistral
ollama serve
```

### 4. Add documents

```bash
mkdir -p data
cp your_documents.pdf data/
```

### 5. Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"reset": true}'
```

### 6. Start the API

```bash
uvicorn deva.api.server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Reference

### `POST /api/v1/chat`

**Request:**
```json
{
  "question": "Summarize the financial report",
  "session_id": "session-001",
  "user_id": "prathap",
  "context_hint": ""
}
```

**Response:**
```json
{
  "answer": "The financial report highlights...",
  "intent": "summarize",
  "enhanced_question": "summarize key findings of the financial report",
  "sources": [
    { "source": "report_q3.pdf", "page": 4 }
  ],
  "latency_seconds": 2.341
}
```

### `POST /api/v1/ingest`

```json
{ "reset": true }
```

### `GET /health`

```json
{ "status": "ok", "service": "deva-rag" }
```

> 📖 Interactive docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Evaluation

**1. Build a golden test dataset** at `deva/eval/test_dataset.json`:

```json
[
  {
    "question": "What is the total revenue in Q3?",
    "ground_truth": "The total revenue in Q3 was $4.2 million."
  }
]
```

**2. Run RAGAS evaluation:**

```bash
python -m deva.eval.rag_eval
```

**3. CI threshold check** (fails if metrics drop):

```bash
python -m deva.eval.ci_eval
```

---

## 📁 Project Structure

```
deva/
├── api/
│   ├── server.py           # FastAPI app + lifespan
│   ├── schemas.py          # Pydantic models
│   └── routes/
│       ├── chat.py         # /chat endpoint
│       └── ingest.py       # /ingest endpoint
├── app/
│   ├── main.py             # CLI entrypoint (non-API mode)
│   ├── rag_chain.py        # LCEL RAG pipeline
│   ├── query_layer.py      # Intent + query enhancement
│   ├── retriever.py        # Vector + hybrid retrieval
│   ├── prompts.py          # Intent-aware prompts
│   ├── guardrails.py       # Output safety layer
│   └── observability.py    # Langfuse tracing
├── ingestion/
│   ├── loaders.py          # Document loaders
│   ├── splitter.py         # Chunking
│   └── indexer.py          # Vectorstore management
├── providers/
│   ├── llm.py              # LLM factory (Ollama / Gemini)
│   ├── embeddings.py       # Embeddings factory
│   └── vectorstore/
│       └── factory.py      # Vectorstore factory
├── eval/
│   ├── rag_eval.py         # Full RAGAS evaluation
│   ├── ci_eval.py          # CI threshold check
│   └── test_dataset.json   # Golden Q&A pairs
├── config.py               # Centralized env config
└── logger.py               # Logging factory
```

---

## 🔐 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEVA_LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `DEVA_LLM_MODEL` | `dolphin-mistral` | Model name |
| `DEVA_DATA_DIR` | `./data` | Document folder |
| `DEVA_CHROMA_DIR` | `./deva_cli/storage/chroma_db` | Vector DB path |
| `DEVA_RETRIEVER` | `mmr` | `mmr` or `hybrid` |
| `DEVA_CHUNK_SIZE` | `1000` | Chunk size for splitting |
| `DEVA_CHUNK_OVERLAP` | `200` | Chunk overlap |
| `DEVA_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `DEVA_LOG_TO_FILE` | `false` | Enable file logging |
| `COHERE_API_KEY` | — | Required for `hybrid` retriever |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse monitoring (optional) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse monitoring (optional) |
| `GEMINI_API_KEY` | — | Required when using Gemini LLM |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
