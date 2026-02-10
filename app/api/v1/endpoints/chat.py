# app/api/v1/endpoints/chat.py
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import json

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag.rag_engine import RAGEngine
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

rag_engine = RAGEngine()


@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """Query the RAG system with conversation memory."""
    logger.info(
        f"[Session: {request.session_id}] "
        f"Query: {request.question[:100]}..."
    )

    try:
        result = await rag_engine.query(
            question=request.question,
            session_id=request.session_id,
            document_ids=request.document_ids
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence"),
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/query/stream")
async def chat_query_stream(request: ChatRequest):
    """Stream chat responses with memory."""
    logger.info(
        f"[Session: {request.session_id}] "
        f"Streaming: {request.question[:100]}..."
    )

    async def generate():
        try:
            async for chunk in rag_engine.query_stream(
                question=request.question,
                session_id=request.session_id,
                document_ids=request.document_ids
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    rag_engine.clear_session(session_id)
    return {"message": f"Session {session_id} cleared", "success": True}


@router.delete("/sessions/all")
async def clear_all_sessions():
    """Clear all conversation sessions."""
    rag_engine.clear_all_sessions()
    return {"message": "All sessions cleared", "success": True}