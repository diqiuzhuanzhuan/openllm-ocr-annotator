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
from openllm_ocr_annotator.annotators.base import BaseAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from utils.prompt_manager import PromptManager
from utils.retry import retry_with_backoff
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Maps provider prefix to its expected environment variable name
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
}

_PROVIDER_URL_ENV_VARS = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "gemini": "GEMINI_API_BASE",
    "mistral": "MISTRAL_API_BASE",
    "groq": "GROQ_BASE_URL",
    "cohere": "COHERE_API_BASE",
    "together_ai": "TOGETHER_BASE_URL",
}


class LiteLLMAnnotator(BaseAnnotator):
    """LiteLLM-based image annotator supporting 100+ LLM providers."""

    @classmethod
    def from_config(cls, config: AnnotatorConfig) -> "LiteLLMAnnotator":
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
        name: str = "litellm_annotator",
        model: str = "openai/gpt-4o",
        task: str = "vision_extraction",
        max_tokens: int = 1000,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        prompt_path: Optional[str] = None,
        n: Optional[int] = 1,
    ):
        """Initialize LiteLLM annotator.

        Args:
            api_key: Provider API key. If None, uses the provider-specific env var
            name: Annotator name used for output directory naming
            model: LiteLLM model string, e.g. "anthropic/claude-3-opus-20240229"
            task: Annotation task type ('ocr', 'layout', 'vision_extraction')
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature. None uses the model default
            base_url: Optional custom API base URL
            prompt_path: Path to prompt templates YAML file
            n: Number of samples to generate per image
        """
        self.model = model
        self.name = name
        self.task = task
        self.max_tokens = max_tokens
        self.temperature = temperature
        provider = self._get_provider()
        provider_url_env = _PROVIDER_URL_ENV_VARS.get(provider)
        self.base_url = base_url or (
            os.getenv(provider_url_env) if provider_url_env else None
        )
        self.prompt_manager = PromptManager(prompt_path=prompt_path)
        self.n = n or 1

        # Set the provider-specific env var if api_key was supplied
        if api_key:
            env_var = _PROVIDER_ENV_VARS.get(provider)
            if env_var:
                os.environ[env_var] = api_key
            else:
                # Fallback: set OPENAI_API_KEY which LiteLLM often checks
                os.environ.setdefault("OPENAI_API_KEY", api_key)

    def _get_provider(self) -> str:
        """Extract provider prefix from the model string.

        Examples:
            "anthropic/claude-3-opus-20240229" → "anthropic"
            "gpt-4-vision-preview"             → "openai"
        """
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"

    def _get_prompt_type(self) -> str:
        """Return the prompt template key for this provider."""
        return self._get_provider()

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    def annotate(
        self, image_path: str, variables: Optional[Dict[str, str]] = None
    ) -> dict:
        """Annotate an image using LiteLLM.

        Args:
            image_path: Path to image file
            variables: Optional variables for prompt template

        Returns:
            dict: Annotation results with a 'result' list of response strings
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Get prompts from template manager; fall back to "openai" template
            prompt_type = self._get_prompt_type()
            try:
                prompts = self.prompt_manager.get_prompt(
                    model=prompt_type, task=self.task, variables=variables
                )
            except Exception:
                prompts = self.prompt_manager.get_prompt(
                    model="openai", task=self.task, variables=variables
                )

            # Encode image to base64
            image_b64 = self._encode_image(image_path)

            messages = [
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

            request_kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.base_url:
                request_kwargs["base_url"] = self.base_url

            # Loop n times — safer than relying on provider-specific n= support
            results = []
            import litellm  # noqa: PLC0415 — lazy import avoids openai version conflicts

            for _ in range(self.n):
                response = litellm.completion(**request_kwargs)
                results.append(response.choices[0].message.content)

            return {
                "result": results,
                "model": self.model,
                "task": self.task,
                "image_path": image_path,
            }

        except Exception as e:
            raise Exception(
                f"Error during LiteLLM annotation of {image_path}: {str(e)}"
            )
