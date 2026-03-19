# 🧠 Deva — Local RAG API

A production-grade, local Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and FastAPI.
Chat with your own documents using local or cloud LLMs — with monitoring, evaluation, and guardrails built in.

> No data leaves your machine unless you choose an online LLM or monitoring service.

---

## ✨ Features

- 📂 **Document ingestion** — PDF, DOCX, TXT support with smart chunking
- 🧠 **Local vector database** — ChromaDB with persistent embeddings
- 🔍 **Hybrid retrieval** — MMR vector search + optional Cohere reranking
- 🎯 **Query intelligence layer** — intent classification + query enhancement before retrieval
- 🤖 **Pluggable LLM support** — Local (Ollama) or Online (Gemini)
- 🔁 **LangGraph orchestration** — graph-based pipeline with extensible nodes
- 📡 **FastAPI REST API** — `/chat`, `/ingest`, and `/health` endpoints
- 📊 **Langfuse monitoring** — full trace observability per session and user
- 🧪 **RAGAS evaluation** — faithfulness, answer relevancy, context precision & recall
- 🛡️ **Guardrails** — prompt-level safety + output enforcement
- 📝 **Structured logging** — levelled, rotating file + stdout logs

---

## 🏗️ Architecture

