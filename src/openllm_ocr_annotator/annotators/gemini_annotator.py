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

import os
import time
from typing import Optional, Dict
from PIL import Image
from google import genai
from src.openllm_ocr_annotator.annotators.base import BaseAnnotator
from utils.prompt_manager import PromptManager
from google.genai import types
import httpx
from utils.logger import setup_logger
from src.openllm_ocr_annotator.config import AnnotatorConfig

logger = setup_logger(__name__)

class GeminiAnnotator(BaseAnnotator):
    """Google Gemini based image annotator."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig):
        """Create annotator instance from config."""
        return GeminiAnnotator(
            name=config.name,
            api_key=config.api_key, 
            model=config.model,
            task=config.task,
            max_tokens=config.max_tokens,
            base_url=config.base_url,
            prompt_path=config.prompt_path,
        )        

    def __init__(
        self, 
        api_key: Optional[str] = None,
        name: str = "gemini_annotator",
        model: str = "gemini-2.5-pro-preview-05-06",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
        base_url: str | httpx.URL | None = None,
        prompt_path: str | None = None,
    ):
        """Initialize OpenAI annotator.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
            model: Model to use for vision tasks
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.base_url = base_url
        if self.base_url:
            logger.warning(f"Warning: Using custom OpenAI API endpoint: {self.base_url}")
        self.client = genai.Client(api_key=self.api_key, 
                                   http_options=types.HttpOptions(base_url=self.base_url))
        self.model = model
        self.task = task
        self.name = name
        self.max_tokens = max_tokens
        self.prompt_manager = PromptManager(prompt_path=prompt_path)

    def annotate(
        self,
        image_path: str,
        variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using Gemini's vision model.
        
        Args:
            image_path: Path to image file
            variables: Optional variables for prompt template
            
        Returns:
            dict: Annotation results
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Get prompts from template manager
            prompts = self.prompt_manager.get_prompt(
                model="gemini",
                task=self.task,
                variables=variables
            )

            # Load and prepare image
            image = Image.open(image_path)

            # Create API request
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role='system',
                        parts=[types.Part.from_text(text=prompts["system"])]
                    ),
                    types.Content(
                        role='user',
                        parts=[types.Part.from_text(text=prompts["user"])],
                    ),
                    image,
                ],
                config={
                    "max_output_tokens": self.max_tokens,
                }
            )

            # Get first response
            response = response.candidates[0]

            return {
                "result": response.text,
                "model": self.model,
                "task": self.task,
                # Gemini doesn't provide timestamp
                "timestamp": int(time.time()),
                "image_path": image_path,
                "safety_ratings": [
                    {
                        "category": rating.category,
                        "probability": rating.probability
                    }
                    for rating in response.safety_ratings
                ] if hasattr(response, "safety_ratings") else []
            }

        except Exception as e:
            raise Exception(f"Error during Gemini annotation: {str(e)}")


