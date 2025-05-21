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
import logging

logger = logging.getLogger(__name__)

class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, image_path: str) -> dict:
        pass

    def _encode_image(self, image_path: str, maximum_size: int = 20*1024*1024) -> str:
        """Encode image to base64 string with automatic resizing if needed.
        
        Args:
            image_path: Path to the image file
            maximum_size: Maximum allowed file size in bytes (default: 20MB)
            
        Returns:
            Base64 encoded string of the image
        """
        # Get original file size
        file_size = os.path.getsize(image_path)
        
        # If file is within size limit, encode directly
        if file_size <= maximum_size:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        
        # If file is too large, resize it
        logger.info(f"Image {image_path} exceeds size limit ({file_size} > {maximum_size}), resizing...")
        
        # Open the image and calculate resize ratio
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
                
            # Start with original size
            width, height = img.size
            ratio = 1.0
            
            # Create a BytesIO object to check size
            while True:
                buffer = BytesIO()
                # Apply current ratio
                new_width = max(int(width * ratio), 1)
                new_height = max(int(height * ratio), 1)
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized.save(buffer, format='JPEG', quality=85, optimize=True)
                
                # Check if size is now within limit
                if buffer.tell() <= maximum_size:
                    break
                    
                # Reduce ratio and try again
                ratio *= 0.75
            
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")