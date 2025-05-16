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
from typing import Optional, Dict
from openai import OpenAI
from .base import BaseAnnotator
from ....utils.prompt_manager import PromptManager

class OpenAIAnnotator(BaseAnnotator):
    """OpenAI GPT-4V based image annotator."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4-vision-preview",
        task: str = "vision_extraction",
        max_tokens: int = 1000
    ):
        """Initialize OpenAI annotator.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
            model: Model to use for vision tasks
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.prompt_manager = PromptManager()
    
    def annotate(
        self, 
        image_path: str,
        variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using OpenAI's vision model.
        
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
                model="openai",
                task=self.task,
                variables=variables
            )
            
            # Encode image
            image_b64 = self._encode_image(image_path)
            
            # Create API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompts["system"]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image_data": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompts["user"]
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            return {
                "result": response.choices[0].message.content,
                "model": self.model,
                "task": self.task,
                "timestamp": response.created,
                "image_path": image_path,
            }
            
        except Exception as e:
            raise Exception(f"Error during OpenAI annotation: {str(e)}")