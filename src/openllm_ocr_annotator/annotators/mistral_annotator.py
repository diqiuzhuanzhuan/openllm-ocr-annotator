# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
from typing import Dict, Optional

from mistralai import Mistral

from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager
from openllm_ocr_annotator.utils.retry import retry_with_backoff

logger = setup_logger(__name__)

MAX_IMAGE_SIZE = 20 * 1024 * 1024


class MistralAnnotator(BaseAnnotator):
    """Mistral vision-model image annotator."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "MistralAnnotator":
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
        name: str = "mistral_annotator",
        model: Optional[str] = "pixtral-large-latest",
        task: str = "vision_extraction",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize a Mistral annotator."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key must be provided or set in MISTRAL_API_KEY "
                "environment variable"
            )

        self.base_url = (
            base_url or os.getenv("MISTRAL_BASE_URL") or os.getenv("MISTRAL_API_BASE")
        )
        if self.base_url:
            logger.warning("Using custom Mistral API endpoint: %s", self.base_url)

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["server_url"] = self.base_url
        self.client = Mistral(**client_kwargs)

        self.name = name
        self.model = model or "pixtral-large-latest"
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
        """Annotate an image using a Mistral vision model."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            prompts = self.prompt_manager.get_prompt(
                model="mistral", task=self.task, variables=variables
            )
            image_b64 = self._encode_image(
                image_path,
                maximum_size=MAX_IMAGE_SIZE,
            )
            image_url = f"data:{self._media_type(image_path)};base64,{image_b64}"

            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompts["system"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": image_url},
                            {"type": "text", "text": prompts["user"]},
                        ],
                    },
                ],
                "n": self.n,
            }
            if self.max_tokens is not None:
                request_kwargs["max_tokens"] = self.max_tokens
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature

            response = self.client.chat.complete(**request_kwargs)
            return {
                "result": [choice.message.content for choice in response.choices],
                "model": self.model,
                "task": self.task,
                "timestamp": response.created,
                "image_path": image_path,
            }
        except Exception as e:
            raise Exception(f"Error during Mistral annotation of {image_path}: {e}")
