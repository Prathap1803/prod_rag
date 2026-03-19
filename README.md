# 🧠 Deva — Local RAG AI Assistant (CLI)

Deva is a **personal, local Retrieval-Augmented Generation (RAG) CLI tool** that lets you chat with your own documents using modern LLMs — fully offline or with optional online models.

It ingests documents, builds a vector database locally, and answers questions **only from your data**.

No cloud required. No data leaves your machine (unless you choose an online LLM).

---

## ✨ Features

- 📂 Ingest documents from a folder (`PDF`, `DOCX`, `TXT`)
- 🧩 Smart document chunking for embeddings
- 🧠 Local vector database using **Chroma**
- 🔍 RAG-based question answering
- 🤖 Pluggable LLM support:
  - Local: **Ollama**
  - Online: **Gemini** (extensible)
- 🔁 Incremental ingestion or full reset
- 🖥️ Simple, fast CLI interface

---

## 🏗️ Architecture Overview

Deva is modular by design:

- **CLI layer** → user interaction
- **Ingestion layer** → load & chunk documents
- **Vector store** → persistent embeddings
- **RAG pipeline** → retrieval + prompt + LLM
- **Providers layer** → swap LLMs & embeddings

📘 See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for details.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/deva-cli.git
cd deva-cli
pip install -e .
```

### 2. (Optional) Install Ollama
```
https://ollama.com
ollama pull dolphin-mistral

```
### 3. Ingest Documents
```
deva ingest ./data
```

### 4. Start Chat
```
deva chat
```
## 🔐 Environment Configuration

Create a .env file:
```
DEVA_LLM_PROVIDER=ollama
DEVA_LLM_MODEL=dolphin-mistral
```

For Gemini:
```
DEVA_LLM_PROVIDER=gemini
DEVA_LLM_MODEL=gemini-1.5-pro
GEMINI_API_KEY=your_api_key_here
```
## 📦 Tech Stack

- LangChain

- ChromaDB

- Ollama

- HuggingFace Sentence Transformers

- Typer

### 🛣️ Roadmap

- [ ] Streaming responses

- [ ] Multiple vector store backends

- [ ] Agent + tool calling

- [ ] Web UI

- [ ] Plugin system