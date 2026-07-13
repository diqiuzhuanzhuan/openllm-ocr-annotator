# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import pytest
import yaml

from openllm_ocr_annotator.utils.prompt_manager import PromptManager


@pytest.fixture
def prompt_file(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "variables": {"name": "default"},
                "default": {
                    "extract": {"system": "default {{name}}", "user": "default"}
                },
                "curator": {
                    "extract": {"system": "curator {{name}}", "user": "typed"}
                },
            }
        )
    )
    return path


def test_get_prompt_selects_by_annotator_type_and_task(prompt_file):
    prompt = PromptManager(prompt_file).get_prompt(
        annotator_type="curator", task="extract", variables={"name": "custom"}
    )

    assert prompt == {"system": "curator custom", "user": "typed"}


def test_get_prompt_falls_back_to_default_type(prompt_file):
    prompt = PromptManager(prompt_file).get_prompt(
        annotator_type="another_type", task="extract"
    )

    assert prompt == {"system": "default default", "user": "default"}


def test_get_prompt_reports_type_and_task_when_missing(prompt_file):
    with pytest.raises(
        ValueError, match="annotator_type=curator, task=missing"
    ):
        PromptManager(prompt_file).get_prompt(
            annotator_type="curator", task="missing"
        )
