# app/utils/helpers.py
import os
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import UploadFile, HTTPException, status

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
        
    Raises:
        HTTPException: If filename is None or file cannot be saved
    """
    # Validate filename exists
    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload"
        )
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    # Get filename (now guaranteed to be str)
    filename: str = upload_file.filename
    
    # Generate unique filename
    file_path = upload_dir / filename
    
    # Handle duplicate filenames
    counter = 1
    while file_path.exists():
        name = Path(filename).stem
        ext = Path(filename).suffix
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


def delete_file(file_path: str) -> None:
    """
    Delete file from disk.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(file_path)