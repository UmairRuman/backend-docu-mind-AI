# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    """Chat request from user."""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search in")
    stream: bool = Field(default=False, description="Enable streaming response")


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


class ChatStreamChunk(BaseModel):
    """Streaming chunk."""
    type: str  # "token", "source", "done"
    content: str
    sources: Optional[List[SourceDocument]] = None