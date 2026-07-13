# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import os
from typing import Dict, Optional

from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager

logger = setup_logger(__name__)

try:
    from bespokelabs import curator
except (
    ImportError
):  # pragma: no cover - exercised only when optional dependency is absent
    curator = None

_CuratorLLMBase = curator.LLM if curator is not None else object


class CuratorAnnotator(BaseAnnotator, _CuratorLLMBase):
    """Annotator backed by Bespoke Labs Curator's LLM request processor."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "CuratorAnnotator":
        return cls(
            name=config.name,
            api_key=config.api_key,
            model=config.model,
            annotator_type=config.type,
            task=config.task,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            base_url=config.base_url,
            prompt_path=config.prompt_path,
            n=config.num_samples,
            tpm=getattr(config, "tpm", None),
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        name: str = "curator_annotator",
        model: Optional[str] = "gpt-4o-mini",
        annotator_type: str = "curator",
        task: str = "vision_extraction",
        max_tokens: Optional[int] = 4096,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
        tpm: Optional[int] = None,
        working_dir: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        if curator is None:
            raise ImportError(
                "CuratorAnnotator requires the optional 'bespokelabs-curator' "
                "package. Install it in an environment whose dependencies do not "
                "conflict with this project's pinned LiteLLM version."
            )

        self.api_key = api_key
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.name = name
        self.model = model or "gpt-4o-mini"
        self.annotator_type = annotator_type
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n or 1
        self.working_dir = working_dir

        self._configure_provider_environment(api_key)

        generation_params = {}
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        if temperature is not None:
            generation_params["temperature"] = temperature

        backend_params = {}
        if self.base_url:
            backend_params["base_url"] = self.base_url
        if tpm:
            backend_params["max_tokens_per_minute"] = tpm

        curator.LLM.__init__(
            self,
            model_name=self.model,
            batch=False,
            backend=backend,
            generation_params=generation_params,
            backend_params=backend_params or None,
        )

    def _configure_provider_environment(self, api_key: Optional[str]) -> None:
        if not api_key:
            return

        provider = self.model.split("/", 1)[0] if "/" in self.model else "openai"
        env_by_provider = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "cohere": "COHERE_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
        }
        os.environ[env_by_provider.get(provider, "OPENAI_API_KEY")] = api_key

    def prompt(self, input: dict):
        variables = input.get("variables") or None
        prompts = self.prompt_manager.get_prompt(
            annotator_type=self.annotator_type,
            task=self.task,
            variables=variables,
        )
        image_b64 = self._encode_image(input["image_path"])
        return [
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompts["user"]},
                ],
            },
        ]

    def parse(self, input: dict, response) -> dict:
        return {"result": response}

    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = []
        for _ in range(self.n):
            response = self(
                [{"image_path": image_path, "variables": variables or {}}],
                working_dir=self.working_dir,
            )
            row = response.dataset[0]
            results.append(row["result"])

        return {
            "result": results,
            "model": self.model,
            "task": self.task,
            "image_path": image_path,
        }
