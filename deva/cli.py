import typer
from deva.app.main import main as chat_main
from deva.ingestion.loaders import load_documents
from deva.ingestion.splitter import split_documents
from deva.ingestion.indexer import get_or_create_vectorstore, add_documents
from deva.config import DATA_DIR

app = typer.Typer(help="Deva – Local RAG agent using Ollama")

@app.command()
def ingest(data_path: str = DATA_DIR, reset: bool = False):
    """Index documents into the vector database"""
    docs = load_documents(data_path)
    chunks = split_documents(docs)
    vectorstore = get_or_create_vectorstore(reset=reset)
    add_documents(vectorstore, chunks)
    typer.echo("🔥 Vector store rebuilt" if reset else "✅ Documents added")

@app.command()
def chat():
    """Start interactive RAG chat"""
    chat_main()

def run():
    app()
