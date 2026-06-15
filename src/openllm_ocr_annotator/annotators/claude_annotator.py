# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import base64
import os
from typing import Dict, Optional

from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager
from openllm_ocr_annotator.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

MAX_IMAGE_SIZE = 5 * 1024 * 1024


class ClaudeAnnotator(BaseAnnotator):
    """Anthropic Claude image annotator."""

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
        model: Optional[str] = "claude-sonnet-4-6",
        task: str = "vision_extraction",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize a Claude annotator."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in "
                "ANTHROPIC_API_KEY environment variable"
            )

        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        if self.base_url:
            logger.warning("Using custom Anthropic API endpoint: %s", self.base_url)

        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "The anthropic package is missing or incomplete. Reinstall project "
                "dependencies before using ClaudeAnnotator."
            ) from e

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = Anthropic(**client_kwargs)

        self.name = name
        self.model = model or "claude-sonnet-4-6"
        self.task = task
        self.max_tokens = max_tokens or 1000
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n or 1

    @staticmethod
    def _media_type(image_b64: str) -> str:
        header = base64.b64decode(image_b64[:32])
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _response_text(response) -> str:
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using a Claude vision model."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            prompts = self.prompt_manager.get_prompt(
                model="claude", task=self.task, variables=variables
            )
            image_b64 = self._encode_image(
                image_path,
                maximum_size=MAX_IMAGE_SIZE,
            )
            request_kwargs = {
                "model": self.model,
                "system": prompts["system"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": self._media_type(image_b64),
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompts["user"]},
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            results = []
            for _ in range(self.n):
                response = self.client.messages.create(**request_kwargs)
                results.append(self._response_text(response))

            return {
                "result": results,
                "model": self.model,
                "task": self.task,
                "image_path": image_path,
            }
        except Exception as e:
            raise Exception(f"Error during Claude annotation of {image_path}: {e}")
