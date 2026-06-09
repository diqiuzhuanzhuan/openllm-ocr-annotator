# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT


# Placeholder for I/O utilities

from pathlib import Path
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


def get_image_files(
    input_dir: str,
    extensions: Set[str] = {".jpg", ".jpeg", ".png", ".webp"},
    recursive: bool = False,
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
            f
            for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]
    else:
        image_files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

    if not image_files:
        logger.warning(
            f"No images found in {input_dir} with extensions: {', '.join(extensions)}"
        )
    else:
        logger.info(
            f"Found {len(image_files)} image files in {input_dir}"
            f"{' (including subdirectories)' if recursive else ''}"
        )

    # Sort for consistent processing order
    return sorted(image_files)
