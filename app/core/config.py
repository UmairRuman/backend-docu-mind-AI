# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via .env file.
    """
    
    # Application Info
    APP_NAME: str = "DocuMind AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Keys
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "documind-index"
    PINECONE_DIMENSION: int = 768  # Google embedding dimension
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "txt", "docx"]
    UPLOAD_DIR: str = "uploads"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Model Configuration
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-1.5-flash"
    
    # RAG Configuration
    TOP_K_RESULTS: int = 4  # Number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.7
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance.
    This ensures settings are loaded only once.
    """
    return Settings()


# Global settings instance
settings = get_settings()