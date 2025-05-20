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


class GeminiAnnotator(BaseAnnotator):
    """Google Gemini based image annotator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro-vision",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
    ):
        """Initialize Gemini annotator.
        
        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var
            model: Model to use for vision tasks (e.g. gemini-pro-vision)
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key must be provided or set in GOOGLE_API_KEY environment variable")

        self.model = genai.Client(api_key=self.api_key)
        self.model_name = model
        self.task = task
        self.max_tokens = max_tokens
        self.prompt_manager = PromptManager()

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
            response = self.model.models.generate_content(

                model=self.model_name,
                contents=[
                    types.Content(
                        role='system',
                        parts=[types.Part.from_text(text=prompts["system"])]
                    ),
                    types.Content(
                        role='user',
                        parts=[types.Part.from_text(text=prompts["user"]), types.Part.from_image(image=image)]
                    ),
                ],
                config={
                    "max_output_tokens": self.max_tokens,
                }
            )

            # Get first response
            response = response.candidates[0]

            return {
                "result": response.text,
                "model": self.model_name,
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


if __name__ == "__main__":
    # Test with Gemini Pro Vision
    annotator = GeminiAnnotator(
        api_key='your-api-key',  # Replace with your API key
        model="gemini-pro-vision",
        task="vision_extraction",
        max_tokens=1000
    )
    result = annotator.annotate("test_image.jpg")
    print(f"\nResults from gemini-pro-vision:")
    print(result)
