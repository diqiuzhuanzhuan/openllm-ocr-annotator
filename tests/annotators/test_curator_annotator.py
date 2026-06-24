# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import patch

from openllm_ocr_annotator.annotators.curator_annotator import CuratorAnnotator, curator
from openllm_ocr_annotator.config.config_manager import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import create_annotator


def _make_config(**kwargs):
    defaults = dict(
        name="curator_test",
        type="curator",
        task="vision_extraction",
        api_key="test-api-key",
        model="gpt-4o-mini",
        num_samples=1,
        temperature=None,
        max_tokens=4096,
    )
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


class TestFromConfig:
    def test_from_config_creates_annotator(self):
        config = _make_config()
        with patch.object(CuratorAnnotator, "__init__", return_value=None) as mock_init:
            CuratorAnnotator.from_config(config)
            mock_init.assert_called_once_with(
                name="curator_test",
                api_key="test-api-key",
                model="gpt-4o-mini",
                task="vision_extraction",
                max_tokens=4096,
                temperature=None,
                base_url=config.base_url,
                prompt_path=config.prompt_path,
                n=1,
                tpm=None,
            )


class TestCreateAnnotator:
    def test_factory_supports_curator_type(self):
        config = _make_config()
        with patch.object(CuratorAnnotator, "from_config", return_value="annotator") as mock_from_config:
            assert create_annotator(config) == "annotator"
            mock_from_config.assert_called_once_with(config=config)


@pytest.mark.skipif(curator is not None, reason="curator optional dependency is installed")
def test_missing_curator_dependency_error_is_actionable():
    config = _make_config()
    with pytest.raises(ImportError, match="bespokelabs-curator"):
        CuratorAnnotator.from_config(config)
