# app/services/rag/rag_engine.py
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.documents  import Document

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.retriever import VectorStoreService
from app.schemas.chat import SourceDocument

logger = get_logger(__name__)


class RAGEngine:
    """
    Production-grade RAG (Retrieval-Augmented Generation) Engine.
    
    This orchestrates the entire RAG pipeline:
    1. Retrieves relevant documents from vector store
    2. Constructs context-aware prompts
    3. Generates answers using LLM
    4. Returns structured responses with sources
    """
    
    def __init__(self):
        logger.info("Initializing RAG Engine")
        
        # Initialize vector store
        self.vector_store = VectorStoreService()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,  # Lower for more factual responses
            convert_system_message_to_human=True
        )
        
        # Custom prompt template for better responses
        self.prompt_template = self._create_prompt_template()
        
        logger.info("RAG Engine initialized successfully")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create a professional prompt template for the RAG system.
        
        This template ensures:
        - The LLM uses only provided context
        - Responses are accurate and well-structured
        - Citations are included
        """
        template = """You are DocuMind AI, an intelligent document assistant. Your role is to provide accurate, helpful answers based ONLY on the provided context from the user's documents.

Context from documents:
{context}

User Question: {question}

Instructions:
1. Answer the question using ONLY the information from the context above
2. If the context doesn't contain enough information to answer fully, acknowledge this
3. Be concise but comprehensive
4. Use bullet points or numbered lists when appropriate
5. If you're unsure or the context is unclear, say so
6. Never make up information not present in the context

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a coherent context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        formatted_docs = []
        
        for idx, doc in enumerate(documents, 1):
            # Extract metadata
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            # Format document
            formatted_doc = f"""
Document {idx} (Source: {filename}, Page: {page}):
{doc.page_content}
---"""
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    def _extract_source_documents(
        self, 
        documents: List[tuple[Document, float]]
    ) -> List[SourceDocument]:
        """
        Convert retrieved documents to SourceDocument schema.
        
        Args:
            documents: List of (Document, score) tuples
            
        Returns:
            List of SourceDocument objects for API response
        """
        sources = []
        
        for doc, score in documents:
            source = SourceDocument(
                document_id=doc.metadata.get("document_id", "unknown"),
                filename=doc.metadata.get("filename", "unknown"),
                page_number=doc.metadata.get("page"),
                chunk_text=doc.page_content[:200] + "...",  # Preview
                relevance_score=float(score)
            )
            sources.append(source)
        
        return sources
    
    async def query(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main query method for RAG system.
        
        Args:
            question: User's question
            document_ids: Optional list of specific document IDs to search
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Step 1: Retrieve relevant documents
            filter_dict = None
            if document_ids:
                # Search only in specified documents
                filter_dict = {"document_id": {"$in": document_ids}}
            
            retrieved_docs = self.vector_store.similarity_search_with_score(
                query=question,
                k=settings.TOP_K_RESULTS,
                filter=filter_dict
            )
            
            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Step 2: Format context
            docs_only = [doc for doc, score in retrieved_docs]
            context = self._format_documents(docs_only)
            
            # Step 3: Generate answer using LLM
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            logger.info("Generating answer with LLM")
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            answer = response.content
            
            # Step 4: Extract sources
            sources = self._extract_source_documents(retrieved_docs)
            
            # Step 5: Calculate confidence (average of retrieval scores)
            avg_score = sum(score for _, score in retrieved_docs) / len(retrieved_docs)
            confidence = float(avg_score)
            
            logger.info(f"Successfully generated answer with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def query_stream(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream responses token by token for better UX.
        
        Args:
            question: User's question
            document_ids: Optional list of specific document IDs
            
        Yields:
            Dictionary chunks with type and content
        """
        logger.info(f"Processing streaming query: {question[:100]}...")
        
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
                yield {
                    "type": "error",
                    "content": "No relevant documents found"
                }
                return
            
            # Step 2: Format context
            docs_only = [doc for doc, score in retrieved_docs]
            context = self._format_documents(docs_only)
            
            # Step 3: Generate streaming answer
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Stream tokens
            async for chunk in self.llm.astream(prompt):
                yield {
                    "type": "token",
                    "content": chunk.content
                }
            
            # Step 4: Send sources at the end
            sources = self._extract_source_documents(retrieved_docs)
            yield {
                "type": "sources",
                "content": "",
                "sources": [source.model_dump() for source in sources]
            }
            
            # Step 5: Done signal
            yield {
                "type": "done",
                "content": ""
            }
            
        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield {
                "type": "error",
                "content": str(e)
            }
    
    def add_document(self, processed_doc: Dict[str, Any]) -> int:
        """
        Add processed document to vector store.
        
        Args:
            processed_doc: Dictionary from DocumentProcessor
            
        Returns:
            Number of chunks added
        """
        try:
            chunks = processed_doc["chunks"]
            self.vector_store.add_documents(chunks)
            
            logger.info(
                f"Added {len(chunks)} chunks for document "
                f"{processed_doc['document_id']}"
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def delete_document(self, document_id: str):
        """
        Delete document from vector store.
        
        Args:
            document_id: Document ID to delete
        """
        try:
            self.vector_store.delete_by_document_id(document_id)
            logger.info(f"Deleted document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise