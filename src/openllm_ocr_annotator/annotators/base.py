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


from abc import ABC, abstractmethod
import base64
import os
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from src.openllm_ocr_annotator.config.config_manager import AnnotatorConfig
from utils.logger import setup_logger

# allow dealing with large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# adjust the DecompressionBomb threshold
Image.MAX_IMAGE_PIXELS = 178956970  # set the same limit as the max pixels size

logger = setup_logger(__name__)


class BaseAnnotator(ABC):
    def _handle_large_image(
        self, image_path: str, max_pixels: int = 178956970
    ) -> Image:
        """Safely open and handle potentially large images.

        Args:
            image_path: Path to the image file
            max_pixels: Maximum allowed total pixels (default: 178956970)

        Returns:
            PIL.Image: Loaded and potentially resized image
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            total_pixels = width * height

            if total_pixels > max_pixels:
                # Calculate new dimensions while maintaining aspect ratio
                ratio = (max_pixels / total_pixels) ** 0.5
                new_width = int(width * ratio)
                new_height = int(height * ratio)

                logger.warning(
                    f"Image {image_path} exceeds pixel limit "
                    f"({total_pixels} > {max_pixels}). "
                    f"Resizing from {width}x{height} to {new_width}x{new_height}"
                )

                return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return img

        except Image.DecompressionBombError as e:
            logger.warning(f"DecompressionBomb warning for {image_path}: {e}")
            # Set a more conservative Image.MAX_IMAGE_PIXELS temporarily
            original_max = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = max_pixels
            try:
                img = Image.open(image_path)
                return img
            finally:
                # Restore original MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = original_max
        except Exception as e:
            raise ValueError(f"Error opening image {image_path}: {e}")

    @staticmethod
    @abstractmethod
    def from_config(cls, config: AnnotatorConfig):
        """Create an annotator instance from a configuration dictionary."""
        pass

    @abstractmethod
    def annotate(self, image_path: str) -> dict:
        pass

    def _encode_image(
        self,
        image_path: str,
        maximum_size: int = 20 * 1024 * 1024,
        max_pixels: int = 178956970,
    ) -> str:
        """Encode image to base64 string with automatic resizing if needed.

        Args:
            image_path: Path to the image file
            maximum_size: Maximum allowed file size in bytes (default: 20MB)
            max_pixels: Maximum allowed total pixels (width * height) (default: 178956970)

        Returns:
            Base64 encoded string of the image
        """
        # First check file size and pixels
        file_size = os.path.getsize(image_path)

        # hander for large images
        img = self._handle_large_image(image_path, max_pixels=max_pixels)

        width, height = img.size
        total_pixels = width * height

        # Check if both limits are satisfied
        if file_size <= maximum_size and total_pixels <= max_pixels:
            # If within limits, encode directly
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # If image needs resizing, log the reason
        if file_size > maximum_size:
            logger.info(
                f"Image {image_path} exceeds size limit ({file_size} > {maximum_size})"
            )
        if total_pixels > max_pixels:
            logger.info(
                f"Image {image_path} exceeds pixel limit ({total_pixels} > {max_pixels})"
            )
        logger.info("Resizing image...")

        # Open the image and calculate resize ratio
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Calculate initial ratio based on both limits
            size_ratio = (
                (maximum_size / file_size) ** 0.5 if file_size > maximum_size else 1.0
            )
            pixel_ratio = (
                (max_pixels / total_pixels) ** 0.5 if total_pixels > max_pixels else 1.0
            )
            ratio = min(size_ratio, pixel_ratio)

            # Create a BytesIO object to check size
            while True:
                buffer = BytesIO()
                # Apply current ratio
                new_width = max(int(width * ratio), 1)
                new_height = max(int(height * ratio), 1)
                new_pixels = new_width * new_height

                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized.save(buffer, format="JPEG", quality=85, optimize=True)

                # Check if both limits are satisfied
                current_size = buffer.tell()
                if current_size <= maximum_size and new_pixels <= max_pixels:
                    break

                # Calculate new ratio based on which limit is exceeded more
                size_exceed_ratio = (
                    current_size / maximum_size if current_size > maximum_size else 1.0
                )
                pixel_exceed_ratio = (
                    new_pixels / max_pixels if new_pixels > max_pixels else 1.0
                )
                exceed_ratio = max(size_exceed_ratio, pixel_exceed_ratio)
                ratio *= (1 / exceed_ratio) ** 0.5

            logger.info(
                f"Resized image from {width}x{height} to {new_width}x{new_height}"
            )
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
