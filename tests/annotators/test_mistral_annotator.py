# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pytest

from openllm_ocr_annotator.annotators.mistral_annotator import MistralAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import create_annotator


def _make_config(**kwargs):
    defaults = {
        "name": "mistral_test",
        "type": "mistral",
        "task": "vision_extraction",
        "api_key": "test-api-key",
        "model": "pixtral-large-latest",
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
    config = _make_config(num_samples=2, base_url="https://mistral.example/v1")

    with patch.object(MistralAnnotator, "__init__", return_value=None) as init:
        MistralAnnotator.from_config(config)

    init.assert_called_once_with(
        name="mistral_test",
        api_key="test-api-key",
        model="pixtral-large-latest",
        task="vision_extraction",
        max_tokens=1200,
        temperature=None,
        base_url="https://mistral.example/v1",
        prompt_path=None,
        n=2,
    )


def test_init_uses_environment_and_custom_base_url(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://mistral.example/v1")

    with (
        patch("openllm_ocr_annotator.annotators.mistral_annotator.Mistral") as mistral,
        patch("openllm_ocr_annotator.annotators.mistral_annotator.PromptManager"),
    ):
        annotator = MistralAnnotator()

    mistral.assert_called_once_with(
        api_key="environment-key",
        server_url="https://mistral.example/v1",
    )
    assert annotator.base_url == "https://mistral.example/v1"


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralAnnotator()


def test_annotate_uses_chat_api_and_returns_samples(mock_image):
    with (
        patch("openllm_ocr_annotator.annotators.mistral_annotator.Mistral") as mistral,
        patch("openllm_ocr_annotator.annotators.mistral_annotator.PromptManager"),
    ):
        annotator = MistralAnnotator(
            api_key="test-key",
            model="pixtral-large-latest",
            max_tokens=1200,
            temperature=0.2,
            n=2,
        )

    annotator.prompt_manager.get_prompt.return_value = {
        "system": "system prompt",
        "user": "user prompt",
    }
    response = MagicMock(created=123456)
    response.choices = [
        MagicMock(message=MagicMock(content="first")),
        MagicMock(message=MagicMock(content="second")),
    ]
    mistral.return_value.chat.complete.return_value = response

    with patch.object(MistralAnnotator, "_encode_image", return_value="base64data"):
        result = annotator.annotate(mock_image)

    assert result["result"] == ["first", "second"]
    assert result["model"] == "pixtral-large-latest"
    assert result["task"] == "vision_extraction"
    assert result["timestamp"] == 123456
    assert result["image_path"] == mock_image

    request = mistral.return_value.chat.complete.call_args.kwargs
    assert request["n"] == 2
    assert request["max_tokens"] == 1200
    assert request["temperature"] == 0.2
    assert request["messages"][1]["content"][0] == {
        "type": "image_url",
        "image_url": "data:image/jpeg;base64,base64data",
    }


def test_annotate_missing_image_returns_error():
    with (
        patch("openllm_ocr_annotator.annotators.mistral_annotator.Mistral"),
        patch("openllm_ocr_annotator.annotators.mistral_annotator.PromptManager"),
        patch("openllm_ocr_annotator.utils.retry.time.sleep"),
    ):
        annotator = MistralAnnotator(api_key="test-key")
        result = annotator.annotate("/nonexistent/image.jpg")

    assert result["status"] == "error"
    assert "Image not found" in result["message"]


def test_factory_creates_mistral_annotator():
    config = _make_config()

    with patch.object(MistralAnnotator, "from_config") as from_config:
        create_annotator(config)

    from_config.assert_called_once_with(config=config)
