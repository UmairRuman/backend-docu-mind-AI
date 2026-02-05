# app/services/rag/embeddings.py
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Google's Generative AI.
    Uses the latest text-embedding-004 model (free tier).
    """
    
    def __init__(self):
        logger.info(f"Initializing Google Embeddings with model: {settings.EMBEDDING_MODEL}")
        
        # Configure Google API
        # genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize embeddings with the latest model
        # FIX: Convert string to SecretStr
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=SecretStr(settings.GOOGLE_API_KEY),  # Convert to SecretStr
            task_type="retrieval_document"
        )
        
        logger.info(f"Embeddings configured for {settings.PINECONE_DIMENSION} dimensions")
        logger.info("Using Google's latest text-embedding-004 model (Free Tier)")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (768-dimensional)
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector (768-dimensional)
        """
        try:
            logger.info(f"Generating embedding for query: {text[:50]}...")
            embedding = self.embeddings.embed_query(text)
            logger.info("Successfully generated query embedding")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise