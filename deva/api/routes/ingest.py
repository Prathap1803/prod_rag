from fastapi import APIRouter, HTTPException
from deva.api.schemas import IngestRequest, IngestResponse
from deva.ingestion.loaders import load_documents
from deva.ingestion.splitter import split_documents
from deva.ingestion.indexer import get_or_create_vectorstore, add_documents
from deva.config import DATA_DIR

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(req: IngestRequest):
    data_path = req.data_path or DATA_DIR

    try:
        docs = load_documents(data_path)
        chunks = split_documents(docs)
        vectorstore = get_or_create_vectorstore(reset=req.reset)
        add_documents(vectorstore, chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        message="✅ Vector store rebuilt" if req.reset else "✅ Documents added",
        chunks_indexed=len(chunks),
    )
