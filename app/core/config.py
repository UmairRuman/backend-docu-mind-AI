# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    
    # API Keys (REQUIRED - must be in .env)
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "documind-index"
    PINECONE_DIMENSION: int = 768
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = "pdf,txt,docx"  # Changed to string
    UPLOAD_DIR: str = "uploads"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Model Configuration
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-1.5-flash"
    
    # RAG Configuration
    TOP_K_RESULTS: int = 4
    SIMILARITY_THRESHOLD: float = 0.7
    
    # CORS Settings
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"  # Changed to string
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Property to get list of allowed extensions
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse comma-separated string into list."""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    # Property to get list of CORS origins
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated string into list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance.
    This ensures settings are loaded only once.
    """
    return Settings()  # type: ignore


# Global settings instance
settings = get_settings()