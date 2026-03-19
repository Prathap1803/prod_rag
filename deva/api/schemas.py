from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_hint: Optional[str] = ""


class SourceDoc(BaseModel):
    source: str
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    intent: str
    enhanced_question: str
    sources: List[SourceDoc]
    latency_seconds: float


class IngestRequest(BaseModel):
    data_path: Optional[str] = None   # defaults to DEVA_DATA_DIR
    reset: bool = False


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int
