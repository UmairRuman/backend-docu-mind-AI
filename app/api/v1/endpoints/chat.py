# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import json

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag.rag_engine import RAGEngine
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize RAG engine
rag_engine = RAGEngine()


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Query the RAG system with a question.
    
    Returns structured response with answer and sources.
    """
    logger.info(f"Received chat query: {request.question[:100]}...")
    
    try:
        result = await rag_engine.query(
            question=request.question,
            document_ids=request.document_ids
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence")
        )
        
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/query/stream")
async def chat_query_stream(request: ChatRequest):
    """
    Stream chat responses token by token.
    
    This provides a better user experience with progressive loading.
    """
    logger.info(f"Received streaming query: {request.question[:100]}...")
    
    async def generate():
        try:
            async for chunk in rag_engine.query_stream(
                question=request.question,
                document_ids=request.document_ids
            ):
                # Send as Server-Sent Events format
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_chunk = {
                "type": "error",
                "content": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )