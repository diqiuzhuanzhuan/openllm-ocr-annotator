# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import base64
import os
from typing import Dict, Optional

import httpx
from openai import OpenAI

from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager
from openllm_ocr_annotator.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_IMAGE_SIZE = 20 * 1024 * 1024


class QwenAnnotator(BaseAnnotator):
    """Qwen vision-model annotator using an OpenAI-compatible endpoint."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "QwenAnnotator":
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
        name: str = "qwen_annotator",
        model: Optional[str] = "qwen-vl-max-latest",
        task: str = "vision_extraction",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = None,
        base_url: str | httpx.URL | None = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize a Qwen annotator."""
        configured_base_url = base_url or os.getenv("QWEN_BASE_URL")
        self.base_url = configured_base_url or DEFAULT_QWEN_BASE_URL
        self.api_key = (
            api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        )
        if not self.api_key and not configured_base_url:
            raise ValueError(
                "Qwen API key must be provided or set in DASHSCOPE_API_KEY "
                "or QWEN_API_KEY"
            )
        if configured_base_url:
            logger.warning("Using custom Qwen API endpoint: %s", self.base_url)

        self.client = OpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
        )
        self.name = name
        self.model = model or "qwen-vl-max-latest"
        self.task = task
        self.max_tokens = max_tokens
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

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using a Qwen vision model."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            prompts = self.prompt_manager.get_prompt(
                model="qwen", task=self.task, variables=variables
            )
            image_b64 = self._encode_image(
                image_path,
                maximum_size=MAX_IMAGE_SIZE,
            )
            image_url = f"data:{self._media_type(image_b64)};base64,{image_b64}"
            messages = [
                {"role": "system", "content": prompts["system"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {"type": "text", "text": prompts["user"]},
                    ],
                },
            ]
            request_kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if self.max_tokens is not None:
                request_kwargs["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            results = []
            timestamp = None
            for _ in range(self.n):
                response = self.client.chat.completions.create(**request_kwargs)
                results.append(response.choices[0].message.content)
                timestamp = response.created

            return {
                "result": results,
                "model": self.model,
                "task": self.task,
                "timestamp": timestamp,
                "image_path": image_path,
            }
        except Exception as e:
            raise Exception(f"Error during Qwen annotation of {image_path}: {e}")
