# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Dict, Optional

import httpx
from openai import OpenAI

from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager
from openllm_ocr_annotator.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
MAX_IMAGE_SIZE = 20 * 1024 * 1024


class GrokAnnotator(BaseAnnotator):
    """xAI Grok image annotator using the OpenAI-compatible Responses API."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "GrokAnnotator":
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
        name: str = "grok_annotator",
        model: Optional[str] = "grok-4.3",
        task: str = "vision_extraction",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = None,
        base_url: str | httpx.URL | None = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize a Grok annotator."""
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "xAI API key must be provided or set in XAI_API_KEY "
                "environment variable"
            )

        self.base_url = base_url or os.getenv("XAI_BASE_URL") or DEFAULT_XAI_BASE_URL
        if str(self.base_url).rstrip("/") != DEFAULT_XAI_BASE_URL:
            logger.warning("Using custom xAI API endpoint: %s", self.base_url)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.name = name
        self.model = model or "grok-4.3"
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n or 1

    @staticmethod
    def _media_type(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        if suffix == ".png" and os.path.getsize(image_path) <= MAX_IMAGE_SIZE:
            return "image/png"
        return "image/jpeg"

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using an image-capable Grok model."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            prompts = self.prompt_manager.get_prompt(
                model="grok", task=self.task, variables=variables
            )
            image_b64 = self._encode_image(
                image_path,
                maximum_size=MAX_IMAGE_SIZE,
            )
            image_url = f"data:{self._media_type(image_path)};base64,{image_b64}"

            request_kwargs = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": prompts["system"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": "high",
                            },
                            {"type": "input_text", "text": prompts["user"]},
                        ],
                    },
                ],
                "store": False,
            }
            if self.max_tokens is not None:
                request_kwargs["max_output_tokens"] = self.max_tokens
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            results = []
            for _ in range(self.n):
                response = self.client.responses.create(**request_kwargs)
                results.append(response.output_text)

            return {
                "result": results,
                "model": self.model,
                "task": self.task,
                "image_path": image_path,
            }
        except Exception as e:
            raise Exception(f"Error during Grok annotation of {image_path}: {e}")
