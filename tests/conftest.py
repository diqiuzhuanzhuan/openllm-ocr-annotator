# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT
import pytest
import yaml
from openllm_ocr_annotator.config.config_manager import AnnotatorConfig


@pytest.fixture
def minimal_annotator_config():
    return AnnotatorConfig(
        name="test_annotator",
        type="openai",
        task="vision_extraction",
        api_key="test-key",
        model="gpt-4-vision-preview",
        weight=1.0,
        num_samples=1,
    )


@pytest.fixture
def sample_annotation():
    return {
        "result": {
            "fields": [
                {"field_name": "invoice_number", "value": "INV-001", "confidence": 0.9},
                {"field_name": "total_amount", "value": "1234.56", "confidence": 0.8},
            ]
        },
        "model": "gpt-4-vision-preview",
        "task": "vision_extraction",
    }


@pytest.fixture
def tmp_yaml_config(tmp_path):
    config = {
        "version": "1.0",
        "task": {
            "task_id": "test_task",
            "input_dir": str(tmp_path / "images"),
            "output_dir": str(tmp_path / "outputs"),
            "prompt_path": str(tmp_path / "prompts.yaml"),
            "max_files": 10,
            "num_samples": 1,
            "annotators": [
                {
                    "name": "annotator_a",
                    "type": "openai",
                    "task": "vision_extraction",
                    "api_key": "key-a",
                    "model": "gpt-4-vision-preview",
                    "weight": 1.0,
                    "enabled": True,
                },
                {
                    "name": "annotator_b",
                    "type": "openai",
                    "task": "vision_extraction",
                    "api_key": "key-b",
                    "model": "gpt-3.5-turbo",
                    "weight": 0.5,
                    "enabled": False,
                },
            ],
            "ensemble": {
                "method": "weighted_vote",
                "min_confidence": 0.0,
                "agreement_threshold": 0.0,
                "enabled": True,
            },
            "dataset": {
                "name": "test_dataset",
                "output_dir": str(tmp_path / "datasets"),
                "split_ratio": 0.8,
                "enabled": False,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path
