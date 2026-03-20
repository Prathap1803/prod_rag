from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deva.api.routes.chat   import router as chat_router
from deva.api.routes.ingest import router as ingest_router
from deva.api.schemas       import HealthResponse
from deva.app.graph         import get_graph
from deva.app.observability import get_langfuse_handler
from deva.logger            import get_logger

load_dotenv()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Deva RAG API...")
    get_langfuse_handler()   # init tracing early (logs warning if keys missing)
    get_graph()              # warm up chain so first request isn't slow
    logger.info("Deva RAG API ready ✅")
    yield
    logger.info("Shutting down Deva RAG API")


app = FastAPI(
    title="Deva RAG API",
    description="Production-grade local RAG system — LangChain + LangGraph + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,   prefix="/api/v1", tags=["Chat"])
app.include_router(ingest_router, prefix="/api/v1", tags=["Ingest"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(status="ok", service="deva-rag")
