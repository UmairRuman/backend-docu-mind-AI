# app/services/rag/retriever.py
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.embeddings import EmbeddingService

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service for managing Pinecone vector store operations.
    """
    
    def __init__(self):
        logger.info("Initializing Pinecone Vector Store")
        
        # Initialize Pinecone (v8 API)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Initialize embeddings with Google's latest model
        self.embedding_service = EmbeddingService()
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Initialize vector store with Google embeddings
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding_service.embeddings
        )
        
        logger.info(f"Vector store initialized with index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [index['name'] for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            
            # Pinecone v8 API
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.PINECONE_DIMENSION,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": settings.PINECONE_ENVIRONMENT
                    }
                }
            )
            logger.info(f"Successfully created index: {self.index_name}")
        else:
            logger.info(f"Index {self.index_name} already exists")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,  # FIX: Use Optional[int]
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Set default if None
        search_k = k if k is not None else settings.TOP_K_RESULTS
        
        try:
            logger.info(f"Searching for top {search_k} similar documents")
            
            results = self.vector_store.similarity_search(
                query=query,
                k=search_k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None,  # FIX: Use Optional[int]
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search with relevance scores."""
        # Set default if None
        search_k = k if k is not None else settings.TOP_K_RESULTS
        
        try:
            logger.info(f"Searching with scores for top {search_k} documents")
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=search_k,
                filter=filter
            )
            
            logger.info(f"Found {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            raise
    
    def delete_by_document_id(self, document_id: str):
        """Delete all chunks belonging to a specific document."""
        try:
            logger.info(f"Deleting document: {document_id}")
            index = self.pc.Index(self.index_name)
            index.delete(filter={"document_id": document_id})
            logger.info(f"Successfully deleted document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise