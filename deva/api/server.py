from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deva.api.routes.chat import router as chat_router
from deva.api.routes.ingest import router as ingest_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the chain on startup so first request isn't slow
    from deva.api.routes.chat import get_chain
    get_chain()
    print("✅ RAG chain warmed up")
    yield
    print("👋 Shutting down Deva API")


app = FastAPI(
    title="Deva RAG API",
    description="Local RAG system powered by LangChain + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest_router, prefix="/api/v1", tags=["Ingest"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "deva-rag"}
