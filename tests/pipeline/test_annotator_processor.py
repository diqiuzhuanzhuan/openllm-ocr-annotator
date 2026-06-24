# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT


import json
import pytest
from unittest.mock import MagicMock, patch
from openllm_ocr_annotator.config.config_manager import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import AnnotatorProcessor


def _make_config(**kwargs):
    defaults = dict(
        name="test_annotator",
        type="openai",
        task="vision_extraction",
        api_key="test-key",
        model="gpt-4-vision-preview",
        num_samples=1,
        temperature=None,
    )
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


@pytest.fixture
def mock_processor(tmp_path):
    """Return an AnnotatorProcessor with its annotator replaced by a mock."""
    config = _make_config()
    with patch(
        "openllm_ocr_annotator.pipeline.annotator_processor.create_annotator"
    ) as mock_create:
        mock_annotator = MagicMock()
        mock_annotator.name = "test_annotator"
        mock_annotator.model = "gpt-4-vision-preview"
        mock_create.return_value = mock_annotator
        processor = AnnotatorProcessor(
            annotator_config=config, output_dir=tmp_path
        )
    return processor


class TestParseAndValidateResult:
    def test_string_input_parsed_as_json(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        result = mock_processor._parse_and_validate_result('{"fields": []}', img)
        assert result is not None
        assert "result" in result

    def test_dict_with_list_result_parsed(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        raw = {"result": ['{"fields": [{"field_name": "k", "value": "v"}]}']}
        result = mock_processor._parse_and_validate_result(raw, img)
        assert result is not None

    def test_empty_result_returns_none(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        raw = {"result": {}}
        result = mock_processor._parse_and_validate_result(raw, img)
        assert result is None

    def test_metadata_added(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        raw = '{"fields": [{"field_name": "k", "value": "v"}]}'
        result = mock_processor._parse_and_validate_result(raw, img)
        assert "metadata" in result
        assert "timestamp" in result["metadata"]


class TestProcessSingleMode:
    def test_uses_cache_when_result_exists(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        # Write a cached result
        cached = {"result": {"fields": [{"field_name": "k", "value": "v"}]}}
        result_path = mock_processor.output_dir / "img.json"
        result_path.write_text(json.dumps(cached))

        mock_processor.annotator.annotate = MagicMock()
        result = mock_processor._process_single_mode(img)

        # annotate should NOT have been called
        mock_processor.annotator.annotate.assert_not_called()
        assert result == cached

    def test_saves_new_result(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()

        mock_processor.annotator.annotate.return_value = {
            "result": ['{"fields": [{"field_name": "k", "value": "v"}]}'],
            "model": "gpt-4-vision-preview",
            "task": "vision_extraction",
            "image_path": str(img),
        }

        result = mock_processor._process_single_mode(img)
        assert result is not None

        result_path = mock_processor.output_dir / "img.json"
        assert result_path.exists()

    def test_annotator_error_returns_none(self, mock_processor, tmp_path):
        img = tmp_path / "img.jpg"
        img.touch()
        mock_processor.annotator.annotate.side_effect = Exception("API error")
        result = mock_processor._process_single_mode(img)
        assert result is None
