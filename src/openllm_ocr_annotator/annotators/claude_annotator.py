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
from anthropic import Anthropic
from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.utils.prompt_manager import PromptManager


class ClaudeAnnotator(BaseAnnotator):
    """Anthropic Claude based image annotator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
    ):
        """Initialize Claude annotator.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var
            model: Model to use for vision tasks (e.g. claude-3-opus-20240229)
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.prompt_manager = PromptManager()

    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using Claude's vision model.

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
                model="claude", task=self.task, variables=variables
            )

            # Encode image
            image_b64 = self._encode_image(image_path)

            # Create API request
            response = self.client.messages.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompts["system"] + "\n" + prompts["user"],
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            return {
                "result": response.content[0].text,
                "model": self.model,
                "task": self.task,
                "timestamp": response.created_at,
                "image_path": image_path,
                "usage": {
                    "prompt_tokens": 0,  # Claude API doesn't provide token counts yet
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        except Exception as e:
            raise Exception(f"Error during Claude annotation: {str(e)}")


if __name__ == "__main__":
    # Test with different Claude models
    models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240229",
    ]

    for model in models:
        annotator = ClaudeAnnotator(
            api_key="sk-ant-api03-...",  # Replace with your API key
            model=model,
            task="vision_extraction",
            max_tokens=1000,
        )
        result = annotator.annotate("test_image.jpg")
        print(f"\nResults from {model}:")
        print(result)
