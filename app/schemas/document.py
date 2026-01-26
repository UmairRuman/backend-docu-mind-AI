# app/schemas/document.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    file_size: int
    status: str = "processing"
    message: str
    chunks_created: Optional[int] = None


class DocumentMetadata(BaseModel):
    """Document metadata."""
    document_id: str
    filename: str
    file_size: int
    upload_date: datetime
    chunks_count: int
    status: str


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentMetadata]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    document_id: str
    message: str
    success: bool