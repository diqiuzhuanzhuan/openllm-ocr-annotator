# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT


import pytest
from openllm_ocr_annotator.config.config_manager import (
    AnnotatorConfig,
    AnnotatorConfigManager,
    EnsembleStrategy,
)


class TestAnnotatorConfigFromDict:
    def test_minimal_dict_uses_defaults(self):
        config = AnnotatorConfig.from_dict(
            {"name": "test", "type": "openai", "task": "vision_extraction"}
        )
        assert config.name == "test"
        assert config.type == "openai"
        assert config.task == "vision_extraction"
        assert config.weight == 1.0
        assert config.enabled is True
        assert config.num_samples == 1
        assert config.api_key is None

    def test_all_fields_set(self):
        config = AnnotatorConfig.from_dict(
            {
                "name": "my_annotator",
                "type": "gemini",
                "task": "ocr",
                "api_key": "secret-key",
                "model": "gemini-pro-vision",
                "base_url": "http://localhost:8080",
                "weight": 0.8,
                "output_format": "json",
                "max_tokens": 500,
                "temperature": 0.7,
                "enabled": False,
                "prompt_path": "/path/to/prompts.yaml",
                "num_samples": 3,
                "backend": "openai",
                "rpm": 500,
                "tpm": 60000,
                "request_timeout": 120,
                "curator_working_dir": "/tmp/curator",
            }
        )
        assert config.name == "my_annotator"
        assert config.type == "gemini"
        assert config.api_key == "secret-key"
        assert config.weight == 0.8
        assert config.enabled is False
        assert config.num_samples == 3
        assert config.temperature == 0.7
        assert config.backend == "openai"
        assert config.rpm == 500
        assert config.tpm == 60000
        assert config.request_timeout == 120
        assert config.curator_working_dir == "/tmp/curator"


class TestEnsembleStrategy:
    def test_from_str_weighted_vote(self):
        strategy = EnsembleStrategy.from_str("weighted_vote")
        assert strategy == EnsembleStrategy.WEIGHTED_VOTE

    def test_from_str_simple_vote(self):
        strategy = EnsembleStrategy.from_str("simple_vote")
        assert strategy == EnsembleStrategy.SIMPLE_VOTE

    def test_from_str_highest_confidence(self):
        strategy = EnsembleStrategy.from_str("highest_confidence")
        assert strategy == EnsembleStrategy.HIGHEST_CONFIDENCE

    def test_from_str_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown voting strategy"):
            EnsembleStrategy.from_str("invalid_strategy")


class TestAnnotatorConfigManager:
    def test_from_file_loads_correctly(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        assert manager.task.task_id == "test_task"
        assert len(manager.task.annotators) == 2

    def test_from_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnnotatorConfigManager.from_file(tmp_path / "nonexistent.yaml")

    def test_get_enabled_annotators(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        enabled = manager.get_enabled_annotators()
        assert len(enabled) == 1
        assert enabled[0].name == "annotator_a"

    def test_get_annotator_weights(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        weights = manager.get_annotator_weights()
        # Only enabled annotators are included
        assert "annotator_a/gpt-4-vision-preview" in weights
        assert weights["annotator_a/gpt-4-vision-preview"] == 1.0

    def test_ensemble_strategy_is_enum(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        assert manager.task.ensemble.method == EnsembleStrategy.WEIGHTED_VOTE

    def test_get_task_config(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        task = manager.get_task_config()
        assert task.max_files == 10
        assert task.num_samples == 1

    def test_get_dataset_config(self, tmp_yaml_config):
        manager = AnnotatorConfigManager.from_file(tmp_yaml_config)
        dataset = manager.get_dataset_config()
        assert dataset.name == "test_dataset"
        assert dataset.split_ratio == 0.8


def test_annotator_config_from_curator_provider_params():
    config = AnnotatorConfig.from_dict(
        {
            "name": "function_call_gpt_4o",
            "type": "curator",
            "task": "function_call_generation",
            "provider": {
                "model_name": "gpt-4o",
                "backend": "openai",
                "backend_params": {
                    "require_all_responses": False,
                    "max_tokens_per_minute": 8_000_000,
                },
                "generation_params": {"max_tokens": 4096},
            },
        }
    )

    assert config.model == "gpt-4o"
    assert config.backend == "openai"
    assert config.tpm == 8_000_000
    assert config.max_tokens == 4096
    assert config.backend_params["require_all_responses"] is False
    assert config.generation_params == {"max_tokens": 4096}


def test_function_call_generation_config_normalizes_to_task(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: "1.0"
function_call_generation:
  enable: true
  function_dataset: "data/tool_query"
  max_num: -1
  output_dir: "data"
  output_format: "jsonl"
  name: "function_call_gpt_4o"
  provider:
    model_name: "gpt-4o"
    backend: "openai"
    backend_params:
      require_all_responses: false
      max_tokens_per_minute: 8_000_000
    generation_params:
      max_tokens: 4096
"""
    )

    manager = AnnotatorConfigManager.from_file(config_path)
    task = manager.get_task_config()
    annotator = manager.get_enabled_annotators()[0]

    assert task.task_id == "function_call_gpt_4o"
    assert task.input_dir == "data/tool_query"
    assert task.max_files == -1
    assert task.ensemble.enabled is False
    assert annotator.type == "curator"
    assert annotator.model == "gpt-4o"
    assert annotator.backend == "openai"
    assert annotator.output_format == "jsonl"
    assert annotator.tpm == 8_000_000
    assert annotator.max_tokens == 4096
