# MIT License
#
# Copyright (c) 2025 LoongMa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pytest
import yaml
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
            }
        )
        assert config.name == "my_annotator"
        assert config.type == "gemini"
        assert config.api_key == "secret-key"
        assert config.weight == 0.8
        assert config.enabled is False
        assert config.num_samples == 3
        assert config.temperature == 0.7


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
