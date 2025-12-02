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
import httpx
from typing import Optional, Dict
from openai import OpenAI
from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.prompt_manager import PromptManager
from openllm_ocr_annotator.utils.retry import retry_with_backoff
from openllm_ocr_annotator.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIAnnotator(BaseAnnotator):
    """OpenAI GPT-4V based image annotator."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig):
        return OpenAIAnnotator(
            name=config.name,
            api_key=config.api_key,
            model=config.model,
            task=config.task,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            base_url=config.base_url,
            prompt_path=config.prompt_path,
            n=config.num_samples,
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: str = "openai_annotator",
        model: str = "gpt-4-vision-preview",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
        temperature: Optional[float] | None = None,
        base_url: str | httpx.URL | None = None,
        prompt_path: str | None = None,
        n: int = None,
    ):
        """Initialize OpenAI annotator.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
            name:
            model: Model to use for vision tasks
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
            temperature:
            base_url:
            prompt_path:
            n:
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.base_url = base_url
        if self.base_url:
            logger.warning(
                f"Warning: Using custom OpenAI API endpoint: {self.base_url}"
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        self.task = task
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
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
                model="openai", task=self.task, variables=variables
            )

            # Encode image
            image_b64 = self._encode_image(image_path, maximum_size=20 * 1024 * 1024)
            # Create API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": prompts["user"]},
                        ],
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=self.n,
            )

            return {
                "result": [
                    response.choices[i].message.content
                    for i in range(len(response.choices))
                ],
                "model": self.model,
                "task": self.task,
                "timestamp": response.created,
                "image_path": image_path,
            }

        except Exception as e:
            raise Exception(
                f"Error during OpenAI annotation annotate {image_path}: {str(e)}"
            )
