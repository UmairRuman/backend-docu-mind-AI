# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Chat request from user."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", description="Session ID for memory")
    document_ids: Optional[List[str]] = Field(None)
    stream: bool = Field(default=False)


class SourceDocument(BaseModel):
    """Source document information."""
    document_id: str
    filename: str
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float


class ChatResponse(BaseModel):
    """Chat response to user."""
    answer: str
    sources: List[SourceDocument]
    confidence: Optional[float] = None
    session_id: str = "default"


class ChatStreamChunk(BaseModel):
    """Streaming chunk."""
    type: str
    content: str
    sources: Optional[List[SourceDocument]] = None