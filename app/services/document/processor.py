# app/services/document/processor.py
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Professional document processor with advanced chunking strategies.
    """
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize text splitter with professional settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Try these in order
            is_separator_regex=False
        )
    
    def generate_document_id(self, filename: str) -> str:
        """Generate unique document ID."""
        timestamp = datetime.now().isoformat()
        unique_string = f"{filename}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def load_document(self, file_path: str, filename: str) -> List[Document]:
        """
        Load document based on file type.
        
        Args:
            file_path: Path to the document
            filename: Original filename
            
        Returns:
            List of LangChain Document objects
        """
        file_extension = Path(filename).suffix.lower()
        
        logger.info(f"Loading document: {filename} (type: {file_extension})")
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            logger.info(f"Successfully loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {filename}: {str(e)}")
            raise
    
    def add_metadata_to_chunks(
        self, 
        chunks: List[Document], 
        document_id: str, 
        filename: str
    ) -> List[Document]:
        """
        Add professional metadata to each chunk for better retrieval.
        
        This includes:
        - Document ID and filename
        - Chunk index and total chunks
        - Original page number (if available)
        - Timestamp
        """
        total_chunks = len(chunks)
        
        enhanced_chunks = []
        for idx, chunk in enumerate(chunks):
            # Preserve existing metadata
            metadata = chunk.metadata.copy() if chunk.metadata else {}
            
            # Add professional metadata
            metadata.update({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "timestamp": datetime.now().isoformat(),
                "chunk_size": len(chunk.page_content)
            })
            
            # Create new document with enhanced metadata
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def process_document(
        self, 
        file_path: str, 
        filename: str
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Steps:
        1. Generate unique document ID
        2. Load document
        3. Split into chunks
        4. Add metadata
        
        Returns:
            Dictionary with document_id and processed chunks
        """
        logger.info(f"Starting document processing: {filename}")
        
        try:
            # Step 1: Generate ID
            document_id = self.generate_document_id(filename)
            
            # Step 2: Load document
            documents = self.load_document(file_path, filename)
            
            # Step 3: Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            
            # Step 4: Add metadata
            enhanced_chunks = self.add_metadata_to_chunks(
                chunks, 
                document_id, 
                filename
            )
            
            logger.info(f"Successfully processed {filename}: {len(enhanced_chunks)} chunks")
            
            return {
                "document_id": document_id,
                "filename": filename,
                "chunks": enhanced_chunks,
                "total_chunks": len(enhanced_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    def validate_file(self, filename: str, file_size: int) -> bool:
        """
        Validate file before processing.
        
        Checks:
        - File extension is allowed
        - File size is within limit
        """
        # Check extension
        file_extension = Path(filename).suffix.lower().replace(".", "")
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type .{file_extension} not allowed. "
                f"Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check size
        if file_size > settings.MAX_UPLOAD_SIZE:
            max_size_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds "
                f"maximum allowed size ({max_size_mb}MB)"
            )
        
        return True