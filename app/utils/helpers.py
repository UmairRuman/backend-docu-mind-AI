# app/utils/helpers.py
import os
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to saved file
    """
    # Create uploads directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_path = upload_dir / upload_file.filename
    
    # Handle duplicate filenames
    counter = 1
    while file_path.exists():
        name = Path(upload_file.filename).stem
        ext = Path(upload_file.filename).suffix
        file_path = upload_dir / f"{name}_{counter}{ext}"
        counter += 1
    
    # Save file
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        logger.info(f"Saved file to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise


def delete_file(file_path: str):
    """Delete file from disk."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)