# app/services/rag/rag_engine.py
from typing import List, Dict, Any, Optional, AsyncIterator, Sequence
from operator import itemgetter
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, SecretStr

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.retriever import VectorStoreService
from app.schemas.chat import SourceDocument

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# In-Memory Chat History Store
# ─────────────────────────────────────────────
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In-memory chat history per session."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


session_store: Dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in session_store:
        logger.info(f"Creating new session: {session_id}")
        session_store[session_id] = InMemoryHistory()
    return session_store[session_id]


class RAGEngine:
    """Production RAG Engine with proper context retrieval."""

    def __init__(self):
        logger.info("Initializing RAG Engine")
        
        self.vector_store = VectorStoreService()
        
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=SecretStr(settings.GOOGLE_API_KEY),
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        self.chain_with_history = self._build_chain()
        logger.info("RAG Engine initialized successfully")

    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant documents found."

        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_docs.append(
                f"[Document {idx}] {filename} (Page {page}):\n{doc.page_content}\n"
            )
        
        return "\n---\n".join(formatted_docs)

    def _build_chain(self) -> RunnableWithMessageHistory:
        """Build LCEL chain with message history."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are DocuMind AI, an intelligent document assistant.

Answer questions based ONLY on the context provided below.

Context:
{context}

Instructions:
- Use ONLY the context above to answer
- Reference conversation history for follow-ups
- If context is insufficient, say "I don't have enough information in the uploaded documents to answer that."
- Be concise and accurate
- Never make up information"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        # Simple chain (retrieval happens in query() method)
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        return RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def _extract_sources(
        self, 
        documents: List[tuple[Document, float]]
    ) -> List[SourceDocument]:
        """Convert retrieved documents to SourceDocument schema."""
        sources = []
        for doc, score in documents:
            sources.append(
                SourceDocument(
                    document_id=doc.metadata.get("document_id", "unknown"),
                    filename=doc.metadata.get("filename", "unknown"),
                    page_number=doc.metadata.get("page"),
                    chunk_text=(
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    ),
                    relevance_score=float(score),
                )
            )
        return sources

    async def query(
        self,
        question: str,
        session_id: str = "default",
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query RAG system with proper context retrieval."""
        logger.info(f"[Session: {session_id}] Query: {question[:100]}...")

        try:
            # ✅ FIX: Actually retrieve documents
            filter_dict = None
            if document_ids:
                filter_dict = {"document_id": {"$in": document_ids}}

            retrieved_docs = self.vector_store.similarity_search_with_score(
                query=question,
                k=settings.TOP_K_RESULTS,
                filter=filter_dict
            )

            if not retrieved_docs:
                logger.warning(f"[Session: {session_id}] No documents found")
                return {
                    "answer": (
                        "I couldn't find relevant information in your documents. "
                        "Please upload documents related to your question."
                    ),
                    "sources": [],
                    "confidence": 0.0,
                }

            # ✅ FIX: Format context properly
            docs_only = [doc for doc, _ in retrieved_docs]
            context = self._format_documents(docs_only)

            logger.info(f"[Session: {session_id}] Retrieved {len(docs_only)} documents")

            # ✅ FIX: Use ainvoke instead of asyncio.to_thread
            answer = await self.chain_with_history.ainvoke(
                {
                    "question": question,
                    "context": context,  # ✅ Actually passing context
                },
                config={"configurable": {"session_id": session_id}}
            )

            sources = self._extract_sources(retrieved_docs)
            avg_score = sum(s for _, s in retrieved_docs) / len(retrieved_docs)

            logger.info(f"[Session: {session_id}] Answer generated")

            return {
                "answer": answer,
                "sources": sources,
                "confidence": float(avg_score),
            }

        except Exception as e:
            logger.error(f"[Session: {session_id}] Query error: {str(e)}", exc_info=True)
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        document_ids: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream responses with proper context."""
        logger.info(f"[Session: {session_id}] Streaming query: {question[:100]}...")

        try:
            # Retrieve documents
            filter_dict = None
            if document_ids:
                filter_dict = {"document_id": {"$in": document_ids}}

            retrieved_docs = self.vector_store.similarity_search_with_score(
                query=question,
                k=settings.TOP_K_RESULTS,
                filter=filter_dict
            )

            if not retrieved_docs:
                yield {
                    "type": "error",
                    "content": "No relevant documents found. Please upload documents first."
                }
                return

            # Format context
            docs_only = [doc for doc, _ in retrieved_docs]
            context = self._format_documents(docs_only)

            # ✅ FIX: Handle streaming properly
            full_answer = ""
            async for chunk in self.chain_with_history.astream(
                {
                    "question": question,
                    "context": context,  # ✅ Actually passing context
                },
                config={"configurable": {"session_id": session_id}}
            ):
                # ✅ FIX: Handle different chunk types
                if isinstance(chunk, str):
                    full_answer += chunk
                    yield {"type": "token", "content": chunk}
                elif isinstance(chunk, AIMessage):
                    content = chunk.content
                    full_answer += "".join(str(item) for item in content)
                    yield {"type": "token", "content": content}

            # Send sources
            sources = self._extract_sources(retrieved_docs)
            yield {
                "type": "sources",
                "content": "",
                "sources": [s.model_dump() for s in sources]
            }

            # ✅ Send completion signal
            yield {"type": "done", "content": ""}
            logger.info(f"[Session: {session_id}] Streaming completed")

        except Exception as e:
            logger.error(f"[Session: {session_id}] Streaming error: {str(e)}", exc_info=True)
            yield {"type": "error", "content": str(e)}

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        if session_id in session_store:
            session_store[session_id].clear()
            logger.info(f"Cleared session: {session_id}")

    def add_document(self, processed_doc: Dict[str, Any]) -> int:
        """Add processed document to vector store."""
        try:
            chunks = processed_doc["chunks"]
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            self.vector_store.add_documents(chunks)
            logger.info(f"Successfully added document {processed_doc['document_id']}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}", exc_info=True)
            raise

    def delete_document(self, document_id: str) -> None:
        """Delete document from vector store."""
        try:
            self.vector_store.delete_by_document_id(document_id)
            logger.info(f"Deleted document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise