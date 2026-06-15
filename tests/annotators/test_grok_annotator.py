# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pytest

from openllm_ocr_annotator.annotators.grok_annotator import (
    DEFAULT_XAI_BASE_URL,
    GrokAnnotator,
)
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import create_annotator


def _make_config(**kwargs):
    defaults = {
        "name": "grok_test",
        "type": "grok",
        "task": "vision_extraction",
        "api_key": "test-api-key",
        "model": "grok-4.3",
        "max_tokens": 1200,
        "temperature": None,
        "num_samples": 1,
    }
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


@pytest.fixture
def mock_image(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    return str(image)


def test_from_config_passes_all_supported_options():
    config = _make_config(num_samples=2, base_url="https://xai.example/v1")

    with patch.object(GrokAnnotator, "__init__", return_value=None) as mock_init:
        GrokAnnotator.from_config(config)

    mock_init.assert_called_once_with(
        name="grok_test",
        api_key="test-api-key",
        model="grok-4.3",
        task="vision_extraction",
        max_tokens=1200,
        temperature=None,
        base_url="https://xai.example/v1",
        prompt_path=None,
        n=2,
    )


def test_init_uses_environment_and_default_base_url(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "environment-key")
    monkeypatch.delenv("XAI_BASE_URL", raising=False)

    with (
        patch("openllm_ocr_annotator.annotators.grok_annotator.OpenAI") as openai,
        patch("openllm_ocr_annotator.annotators.grok_annotator.PromptManager"),
    ):
        annotator = GrokAnnotator()

    openai.assert_called_once_with(
        api_key="environment-key",
        base_url=DEFAULT_XAI_BASE_URL,
    )
    assert annotator.base_url == DEFAULT_XAI_BASE_URL


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        GrokAnnotator()


def test_annotate_uses_responses_api_and_returns_samples(mock_image):
    with (
        patch("openllm_ocr_annotator.annotators.grok_annotator.OpenAI") as openai,
        patch("openllm_ocr_annotator.annotators.grok_annotator.PromptManager"),
    ):
        annotator = GrokAnnotator(
            api_key="test-key",
            model="grok-4.3",
            max_tokens=1200,
            temperature=0.2,
            n=2,
        )

    annotator.prompt_manager.get_prompt.return_value = {
        "system": "system prompt",
        "user": "user prompt",
    }
    openai.return_value.responses.create.side_effect = [
        MagicMock(output_text="first"),
        MagicMock(output_text="second"),
    ]

    with patch.object(GrokAnnotator, "_encode_image", return_value="base64data"):
        result = annotator.annotate(mock_image)

    assert result["result"] == ["first", "second"]
    assert result["model"] == "grok-4.3"
    assert result["task"] == "vision_extraction"
    assert result["image_path"] == mock_image
    assert openai.return_value.responses.create.call_count == 2

    request = openai.return_value.responses.create.call_args.kwargs
    assert request["max_output_tokens"] == 1200
    assert request["temperature"] == 0.2
    assert request["store"] is False
    assert request["input"][1]["content"][0] == {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64,base64data",
        "detail": "high",
    }


def test_annotate_missing_image_returns_error():
    with (
        patch("openllm_ocr_annotator.annotators.grok_annotator.OpenAI"),
        patch("openllm_ocr_annotator.annotators.grok_annotator.PromptManager"),
    ):
        annotator = GrokAnnotator(api_key="test-key")

    result = annotator.annotate("/nonexistent/image.jpg")

    assert result["status"] == "error"
    assert "Image not found" in result["message"]


def test_factory_creates_grok_annotator():
    config = _make_config()

    with patch.object(GrokAnnotator, "from_config") as from_config:
        create_annotator(config)

    from_config.assert_called_once_with(config=config)
