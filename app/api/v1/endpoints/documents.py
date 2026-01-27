# app/api/v1/endpoints/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List

from app.schemas.document import (
    DocumentUploadResponse,
    DocumentDeleteResponse
)
from app.services.document.processor import DocumentProcessor
from app.services.rag.rag_engine import RAGEngine
from app.utils.helpers import save_upload_file, delete_file, get_file_size
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize services
doc_processor = DocumentProcessor()
rag_engine = RAGEngine()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Steps:
    1. Validate file
    2. Save to disk
    3. Process (parse, chunk, embed)
    4. Add to vector store
    5. Clean up temporary file
    """
    # Check if filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    logger.info(f"Received upload request for file: {file.filename}")
    
    try:
        # Step 1: Save file
        file_path = await save_upload_file(file)
        file_size = get_file_size(file_path)
        
        # Step 2: Validate (now filename is guaranteed to be str)
        doc_processor.validate_file(file.filename, file_size)
        
        # Step 3: Process document
        processed_doc = doc_processor.process_document(file_path, file.filename)
        
        # Step 4: Add to vector store
        chunks_added = rag_engine.add_document(processed_doc)
        
        # Step 5: Clean up (optional - you might want to keep files)
        # delete_file(file_path)
        
        logger.info(f"Successfully processed document: {file.filename}")
        
        return DocumentUploadResponse(
            document_id=processed_doc["document_id"],
            filename=file.filename,
            file_size=file_size,
            status="completed",
            message="Document processed successfully",
            chunks_created=chunks_added
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    """
    Delete a document from the vector store.
    """
    logger.info(f"Deleting document: {document_id}")
    
    try:
        rag_engine.delete_document(document_id)
        
        return DocumentDeleteResponse(
            document_id=document_id,
            message="Document deleted successfully",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )