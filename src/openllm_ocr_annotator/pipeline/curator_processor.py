# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.pipeline.image_dataset import (
    build_image_dataset,
    model_output_dir,
    result_path_for_row,
)
from openllm_ocr_annotator.utils.formatter import parse_json_from_text
from openllm_ocr_annotator.utils.logger import setup_logger
from openllm_ocr_annotator.utils.prompt_manager import PromptManager

logger = setup_logger(__name__)

try:
    from bespokelabs import curator
except ImportError:  # pragma: no cover - optional dependency guard
    curator = None


def _optional_config(config: AnnotatorConfig, field: str, default=None):
    value = getattr(config, field, default)
    return default if value is None else value


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _merged_generation_params(config: AnnotatorConfig) -> dict[str, Any]:
    params = dict(getattr(config, "generation_params", None) or {})
    params.setdefault("max_tokens", config.max_tokens)
    params.setdefault("temperature", config.temperature)
    return _clean_dict(params)


def _merged_backend_params(config: AnnotatorConfig) -> dict[str, Any]:
    params = dict(getattr(config, "backend_params", None) or {})
    params.setdefault("base_url", _provider_base_url(config))
    params.setdefault(
        "max_requests_per_minute", _optional_config(config, "rpm", 1_000_000)
    )
    params.setdefault(
        "max_tokens_per_minute", _optional_config(config, "tpm", 10_000_000)
    )
    params.setdefault(
        "request_timeout", _optional_config(config, "request_timeout", 60)
    )
    return _clean_dict(params)


def _enable_aiohttp_proxy_from_env() -> None:
    import aiohttp

    if getattr(aiohttp.ClientSession, "_openllm_trust_env_patched", False):
        return

    original_client_session = aiohttp.ClientSession

    def client_session_with_trust_env(*args, **kwargs):
        kwargs.setdefault("trust_env", True)
        return original_client_session(*args, **kwargs)

    client_session_with_trust_env._openllm_trust_env_patched = True
    aiohttp.ClientSession = client_session_with_trust_env


def _provider_from_model(config: AnnotatorConfig) -> str:
    return (
        config.model.split("/", 1)[0]
        if config.model and "/" in config.model
        else "openai"
    )


def _set_provider_api_key(config: AnnotatorConfig) -> None:
    if not config.api_key:
        return
    provider = _provider_from_model(config)
    env_by_provider = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
    }
    os.environ[env_by_provider.get(provider, "OPENAI_API_KEY")] = config.api_key


def _provider_base_url(config: AnnotatorConfig) -> str | None:
    if config.base_url:
        return config.base_url
    provider = _provider_from_model(config)
    env_by_provider = {
        "openai": "OPENAI_BASE_URL",
        "anthropic": "ANTHROPIC_BASE_URL",
        "gemini": "GEMINI_API_BASE",
        "mistral": "MISTRAL_API_BASE",
        "groq": "GROQ_BASE_URL",
        "cohere": "COHERE_API_BASE",
        "together_ai": "TOGETHER_BASE_URL",
    }
    env_var = env_by_provider.get(provider, "OPENAI_BASE_URL")
    return os.getenv(env_var)


if curator is not None:

    class CuratorVisionLLM(curator.LLM):
        """Curator LLM that converts image dataset rows into multimodal prompts."""

        def __init__(
            self,
            *args,
            prompt_manager: PromptManager,
            annotator_type: str,
            task: str,
            **kwargs,
        ):
            self.prompt_manager = prompt_manager
            self.annotator_type = annotator_type
            self.task = task
            super().__init__(*args, **kwargs)

        def prompt(self, row: dict):
            variables = json.loads(row.get("variables") or "{}")
            prompts = self.prompt_manager.get_prompt(
                annotator_type=self.annotator_type,
                task=self.task,
                variables=variables or None,
            )
            prompt_text = f"{prompts['system']}\n\n{prompts['user']}"
            mime_type, _ = mimetypes.guess_type(row["image_path"])
            mime_type = mime_type or "image/png"
            with open(row["image_path"], "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{image_b64}"
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]

        def parse(self, row: dict, response) -> dict:
            return {
                "stem": row["stem"],
                "filename": row["filename"],
                "image_path": row["image_path"],
                "sample_id": row.get("sample_id", 0),
                "result": response,
            }

else:

    class CuratorVisionLLM:  # pragma: no cover - instantiated only without dependency
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Curator pipeline requires the optional 'bespokelabs-curator' package."
            )


class CuratorAnnotatorProcessor:
    """Dataset-first batch processor backed by curator.LLM."""

    def __init__(
        self,
        config: AnnotatorConfig,
        output_dir: Path,
        task_prompt_path: str | None = None,
        num_samples: int | None = None,
    ):
        if curator is None:
            raise ImportError(
                "Curator pipeline requires the optional 'bespokelabs-curator' package."
            )
        self.config = config
        self.output_dir = output_dir
        self.model_dir = model_output_dir(output_dir, config)
        self.prompt_manager = PromptManager(
            prompt_path=config.prompt_path or task_prompt_path
        )
        effective_num_samples = (
            config.num_samples if num_samples is None else num_samples
        )
        self.num_samples = max(effective_num_samples or 1, 1)

    def build_dataset(self, image_files: Iterable[Path]) -> Dataset:
        return build_image_dataset(
            image_files,
            num_samples=self.num_samples,
            output_dir=self.model_dir,
        )

    def run(self, image_files: Iterable[Path]) -> None:
        dataset = self.build_dataset(image_files)
        self._ensure_output_dirs()
        if len(dataset) == 0:
            logger.info(
                "All curator results already exist for %s/%s",
                self.config.name,
                self.config.model,
            )
            return

        logger.info(
            "Running curator annotator %s/%s on %s request rows",
            self.config.name,
            self.config.model,
            len(dataset),
        )
        _set_provider_api_key(self.config)
        _enable_aiohttp_proxy_from_env()
        llm = self._create_llm()
        response = llm(dataset, working_dir=str(self._curator_working_dir()))
        self._save_response_dataset(response)

    def _ensure_output_dirs(self) -> None:
        if self.num_samples > 1:
            for sample_id in range(self.num_samples):
                (self.model_dir / "sampling" / f"sample_{sample_id}").mkdir(
                    parents=True, exist_ok=True
                )
        else:
            self.model_dir.mkdir(parents=True, exist_ok=True)

    def _curator_working_dir(self) -> Path:
        configured = _optional_config(self.config, "curator_working_dir")
        if configured:
            return Path(configured)
        return (
            self.output_dir
            / ".curator_dataset_v4"
            / self.config.name
            / (self.config.model or "default").replace("/", "__")
        )

    def _create_llm(self) -> CuratorVisionLLM:
        generation_params = _merged_generation_params(self.config)
        backend_params = _merged_backend_params(self.config)
        llm = CuratorVisionLLM(
            model_name=self.config.model,
            backend=_optional_config(self.config, "backend"),
            batch=False,
            generation_params=generation_params,
            backend_params=backend_params or None,
            prompt_manager=self.prompt_manager,
            annotator_type=self.config.type,
            task=self.config.task,
        )
        self._patch_token_estimation(llm)
        return llm

    def _patch_token_estimation(self, llm: CuratorVisionLLM) -> None:
        processor = llm._request_processor
        original_estimate = processor.estimate_total_tokens
        token_count_type = type(original_estimate([{"role": "user", "content": ""}]))
        input_estimate = _optional_config(self.config, "estimated_input_tokens", 1200)
        output_estimate = self.config.max_tokens or 4096

        def estimate_total_tokens(_messages):
            return token_count_type(input=input_estimate, output=output_estimate)

        processor.estimate_total_tokens = estimate_total_tokens
        if hasattr(processor, "estimate_output_tokens"):
            processor.estimate_output_tokens = lambda: output_estimate

    def _save_response_dataset(self, dataset: Dataset) -> None:
        for row in dataset:
            sample_id = row.get("sample_id", 0) if self.num_samples > 1 else None
            save_path = result_path_for_row(self.model_dir, row["stem"], sample_id)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result = self._format_row_result(row, sample_id)
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    def _format_row_result(self, row: dict, sample_id: int | None) -> dict:
        raw_result = row["result"]
        parsed_result = (
            parse_json_from_text(raw_result)
            if isinstance(raw_result, str)
            else raw_result
        )
        return {
            "result": parsed_result or raw_result,
            "model": self.config.model,
            "task": self.config.task,
            "image_path": row["image_path"],
            "metadata": _clean_dict(
                {
                    "timestamp": int(time.time()),
                    "sample_id": sample_id,
                    "temperature": self.config.temperature
                    if sample_id is not None
                    else None,
                }
            ),
        }


def run_curator_annotation(
    annotator_configs: list[AnnotatorConfig],
    output_dir: Path,
    image_files: list[Path],
    task_prompt_path: str | None = None,
    num_samples: int = 1,
) -> None:
    """Run all curator annotators over image files using curator's dataset model."""
    for config in annotator_configs:
        processor = CuratorAnnotatorProcessor(
            config=config,
            output_dir=output_dir,
            task_prompt_path=task_prompt_path,
            num_samples=num_samples,
        )
        processor.run(image_files)
