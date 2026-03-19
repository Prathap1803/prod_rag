from fastapi import APIRouter, HTTPException
from deva.api.schemas import IngestRequest, IngestResponse
from deva.ingestion.loaders import load_documents
from deva.ingestion.splitter import split_documents
from deva.ingestion.indexer import get_or_create_vectorstore, add_documents
from deva.config import DATA_DIR
from deva.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(req: IngestRequest):
    
    data_path = req.data_path or DATA_DIR
    logger.info(f"Ingest request | path={data_path} | reset={req.reset}")
    try:
        docs = load_documents(data_path)
        chunks = split_documents(docs)
        vectorstore = get_or_create_vectorstore(reset=req.reset)
        add_documents(vectorstore, chunks)
        logger.info(f"Ingestion complete | chunks={len(chunks)}")
    except ValueError as e:
        logger.warning(f"Ingestion validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        message="✅ Vector store rebuilt" if req.reset else "✅ Documents added",
        chunks_indexed=len(chunks),
    )
