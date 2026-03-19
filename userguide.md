# 📘 Deva User Guide

This guide helps you install and use Deva locally.

---

## 🖥️ Requirements

- Python 3.10+
- pip
- (Optional) Ollama for local models

---

## 📦 Installation

### Clone Repository

```
git clone https://github.com/yourusername/deva-cli.git
cd deva-cli
```
## Install in Editable Mode

```
pip install -e .
```

## 🤖 LLM Setup
### Option 1: Local (Ollama)
```
Install Ollama: https://ollama.com
```
Pull a model:
```
ollama pull dolphin-mistral
```

Create .env:
```
DEVA_LLM_PROVIDER=ollama
DEVA_LLM_MODEL=dolphin-mistral
```
### Option 2: Gemini (Cloud)

Get an API key from Google AI Studio

Install dependency:
```
pip install langchain-google-genai
```

Configure .env:
```
DEVA_LLM_PROVIDER=gemini
DEVA_LLM_MODEL=gemini-1.5-pro
GEMINI_API_KEY=your_key_here
```
## 📂 Ingest Documents
```
deva ingest ./data
```
Reset Vector Database
```
deva ingest ./data --reset
```

## Supported formats:

- PDF

- DOCX

- TXT

## 💬 Chat with Your Data
```
deva chat
```

Exit with:
```
exit
quit
```
## ⚙️CLI Commands
Command	Description
```
deva ingest PATH	Index documents
deva ingest PATH --reset	Rebuild database
deva chat	Interactive chat
```
## 🧠 Best Practices

- Use clean, text-based PDFs

- Avoid scanned documents

- Reset DB after major document changes

- Keep documents topic-focused

## 🛠️ Troubleshooting
LLM not responding

- Ensure Ollama is running

- Check API keys

Empty answers

- Documents not ingested

- Retriever k too low

- Question outside document scope