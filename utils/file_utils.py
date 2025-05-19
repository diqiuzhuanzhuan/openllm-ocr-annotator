# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Placeholder for I/O utilities

from pathlib import Path
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

def get_image_files(
    input_dir: str,
    extensions: Set[str] = {".jpg", ".jpeg", ".png", ".webp"},
    recursive: bool = False
) -> List[Path]:
    """Get all image files from input directory.
    
    Args:
        input_dir: Directory containing images
        extensions: Set of supported file extensions (default: {'.jpg', '.jpeg', '.png', '.webp'})
        recursive: Whether to search recursively in subdirectories (default: False)
        
    Returns:
        List of Path objects for image files
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If no image files found with supported extensions
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_dir}")
    
    # Get all files with supported extensions
    if recursive:
        image_files = [
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]
    else:
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
    
    if not image_files:
        logger.warning(
            f"No images found in {input_dir} "
            f"with extensions: {', '.join(extensions)}"
        )
    else:
        logger.info(
            f"Found {len(image_files)} image files in {input_dir}"
            f"{' (including subdirectories)' if recursive else ''}"
        )
    
    # Sort for consistent processing order
    return sorted(image_files)