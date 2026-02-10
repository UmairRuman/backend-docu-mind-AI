# app/services/rag/rag_engine.py
from typing import List, Dict, Any, Optional, AsyncIterator , Sequence
from operator import itemgetter
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, SecretStr

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.retriever import VectorStoreService
from app.schemas.chat import SourceDocument

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# In-Memory Chat History Store
# Stores separate history per session_id
# ─────────────────────────────────────────────
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In-memory chat history per session."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


# Global session store - holds history per session_id
session_store: Dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create chat history for a given session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Chat history for that session
    """
    if session_id not in session_store:
        logger.info(f"Creating new session: {session_id}")
        session_store[session_id] = InMemoryHistory()
    return session_store[session_id]


class RAGEngine:
    """
    Production-grade RAG Engine with RunnableWithMessageHistory.
    
    Features:
    - Modern LangChain LCEL pipeline
    - Per-session conversation memory
    - Streaming support
    - Source citations
    """

    def __init__(self):
        logger.info("Initializing RAG Engine with RunnableWithMessageHistory")

        # Initialize vector store
        self.vector_store = VectorStoreService()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=SecretStr(settings.GOOGLE_API_KEY),
            temperature=0.3,
            convert_system_message_to_human=True
        )

        # Build the LCEL chain with history
        self.chain_with_history = self._build_chain()

        logger.info("RAG Engine initialized successfully")

    def _build_chain(self) -> RunnableWithMessageHistory:
        """
        Build the modern LCEL RAG chain with message history.
        
        Chain Flow:
        User Question
            → Retrieve relevant documents
            → Format context
            → Inject into prompt with history
            → LLM generates answer
            → Parse output
        """

        # Prompt template with history support
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are DocuMind AI, an intelligent document assistant.
Your role is to provide accurate, helpful answers based ONLY on the provided context.

Key Instructions:
1. Answer ONLY from the context below
2. Use conversation history for follow-up context
3. If context is insufficient, say so clearly
4. Be concise but thorough
5. Never hallucinate or make up information

Context from documents:
{context}"""
            ),
            # This placeholder injects full conversation history
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Build LCEL chain
        chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | RunnablePassthrough()
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Wrap with history management
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,          # Function to get/create history
            input_messages_key="question", # Key for user input
            history_messages_key="history" # Key for history placeholder
        )

        return chain_with_history

    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant documents found."

        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            formatted_docs.append(
                f"Document {idx} (Source: {filename}, Page: {page}):\n"
                f"{doc.page_content}\n---"
            )

        return "\n".join(formatted_docs)

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
        """
        Query the RAG system with conversation memory.

        Args:
            question: User's question
            session_id: Unique session ID for memory isolation
            document_ids: Optional specific document IDs to search

        Returns:
            Dictionary with answer, sources, and confidence
        """
        logger.info(f"[Session: {session_id}] Query: {question[:100]}...")

        try:
            # Step 1: Retrieve relevant documents
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
                        "Please make sure you've uploaded documents related to your query."
                    ),
                    "sources": [],
                    "confidence": 0.0,
                }

            # Step 2: Format context
            docs_only = [doc for doc, _ in retrieved_docs]
            context = self._format_documents(docs_only)

            # Step 3: Invoke chain with session history
            # RunnableWithMessageHistory handles history automatically
            answer = await asyncio.to_thread(
                self.chain_with_history.invoke,
                {
                    "question": question,
                    "context": context,
                },
                config={"configurable": {"session_id": session_id}}
            )

            # Step 4: Extract sources and confidence
            sources = self._extract_sources(retrieved_docs)
            avg_score = sum(s for _, s in retrieved_docs) / len(retrieved_docs)

            logger.info(
                f"[Session: {session_id}] "
                f"Answer generated with {len(sources)} sources"
            )

            return {
                "answer": answer,
                "sources": sources,
                "confidence": float(avg_score),
            }

        except Exception as e:
            logger.error(
                f"[Session: {session_id}] Query error: {str(e)}",
                exc_info=True
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        document_ids: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream responses with session memory.

        Args:
            question: User's question
            session_id: Session ID for memory isolation
            document_ids: Optional specific document IDs
        """
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

            # Stream tokens with history
            full_answer = ""
            async for chunk in self.chain_with_history.astream(
                {
                    "question": question,
                    "context": context,
                },
                config={"configurable": {"session_id": session_id}}
            ):
                full_answer += chunk
                yield {
                    "type": "token",
                    "content": chunk
                }

            # Send sources after streaming
            sources = self._extract_sources(retrieved_docs)
            yield {
                "type": "sources",
                "content": "",
                "sources": [s.model_dump() for s in sources]
            }

            yield {"type": "done", "content": ""}
            logger.info(f"[Session: {session_id}] Streaming completed")

        except Exception as e:
            logger.error(
                f"[Session: {session_id}] Streaming error: {str(e)}",
                exc_info=True
            )
            yield {"type": "error", "content": str(e)}

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        if session_id in session_store:
            session_store[session_id].clear()
            logger.info(f"Cleared session: {session_id}")

    def clear_all_sessions(self) -> None:
        """Clear all sessions."""
        session_store.clear()
        logger.info("All sessions cleared")

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