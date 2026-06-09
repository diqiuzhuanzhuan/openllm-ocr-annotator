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
from openllm_ocr_annotator.config import AnnotatorConfig
from utils.prompt_manager import PromptManager
from utils.retry import retry_with_backoff
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ClaudeAnnotator(BaseAnnotator):
    """Anthropic Claude based image annotator."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "ClaudeAnnotator":
        return cls(
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
        name: str = "claude_annotator",
        model: str = "claude-3-opus-20240229",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize Claude annotator.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var
            name: Annotator name used for output directory naming
            model: Model to use for vision tasks (e.g. claude-3-opus-20240229)
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature. None uses the model default
            base_url: Optional custom API base URL
            prompt_path: Path to prompt templates YAML file
            n: Number of samples to generate per image
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        self.client = Anthropic(api_key=self.api_key, base_url=self.base_url)
        self.name = name
        self.model = model
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n or 1

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using Claude's vision model.

        Args:
            image_path: Path to image file
            variables: Optional variables for prompt template

        Returns:
            dict: Annotation results with a 'result' list of response strings
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

            # Build API request kwargs
            request_kwargs = dict(
                model=self.model,
                system=prompts["system"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompts["user"]},
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            # Collect n samples
            results = []
            for _ in range(self.n):
                response = self.client.messages.create(**request_kwargs)
                results.append(response.content[0].text)

            return {
                "result": results,
                "model": self.model,
                "task": self.task,
                "image_path": image_path,
            }

        except Exception as e:
            raise Exception(f"Error during Claude annotation of {image_path}: {str(e)}")
