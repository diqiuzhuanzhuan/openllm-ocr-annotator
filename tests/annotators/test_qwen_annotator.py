# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import base64
from unittest.mock import MagicMock, patch

import pytest

from openllm_ocr_annotator.annotators.qwen_annotator import (
    DEFAULT_QWEN_BASE_URL,
    QwenAnnotator,
)
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import create_annotator


def _make_config(**kwargs):
    defaults = {
        "name": "qwen_test",
        "type": "qwen",
        "task": "vision_extraction",
        "api_key": "test-api-key",
        "model": "qwen-vl-max-latest",
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
    config = _make_config(num_samples=2, base_url="http://qwen.local/v1")

    with patch.object(QwenAnnotator, "__init__", return_value=None) as init:
        QwenAnnotator.from_config(config)

    init.assert_called_once_with(
        name="qwen_test",
        api_key="test-api-key",
        model="qwen-vl-max-latest",
        task="vision_extraction",
        max_tokens=1200,
        temperature=None,
        base_url="http://qwen.local/v1",
        prompt_path=None,
        n=2,
    )


def test_init_uses_dashscope_environment_and_default_url(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-key")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)

    with (
        patch("openllm_ocr_annotator.annotators.qwen_annotator.OpenAI") as openai,
        patch("openllm_ocr_annotator.annotators.qwen_annotator.PromptManager"),
    ):
        annotator = QwenAnnotator()

    openai.assert_called_once_with(
        api_key="environment-key",
        base_url=DEFAULT_QWEN_BASE_URL,
    )
    assert annotator.base_url == DEFAULT_QWEN_BASE_URL


def test_init_allows_local_endpoint_without_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    with (
        patch("openllm_ocr_annotator.annotators.qwen_annotator.OpenAI") as openai,
        patch("openllm_ocr_annotator.annotators.qwen_annotator.PromptManager"),
    ):
        QwenAnnotator(base_url="http://qwen.local/v1")

    openai.assert_called_once_with(
        api_key="EMPTY",
        base_url="http://qwen.local/v1",
    )


def test_init_requires_api_key_for_dashscope(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        QwenAnnotator()


def test_media_type_detects_png():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\ncontent").decode()

    assert QwenAnnotator._media_type(encoded) == "image/png"


def test_annotate_uses_chat_api_and_returns_samples(mock_image):
    with (
        patch("openllm_ocr_annotator.annotators.qwen_annotator.OpenAI") as openai,
        patch("openllm_ocr_annotator.annotators.qwen_annotator.PromptManager"),
    ):
        annotator = QwenAnnotator(
            api_key="test-key",
            model="qwen-vl-max-latest",
            max_tokens=1200,
            temperature=0.2,
            n=2,
        )

    annotator.prompt_manager.get_prompt.return_value = {
        "system": "system prompt",
        "user": "user prompt",
    }
    openai.return_value.chat.completions.create.side_effect = [
        MagicMock(
            created=1,
            choices=[MagicMock(message=MagicMock(content="first"))],
        ),
        MagicMock(
            created=2,
            choices=[MagicMock(message=MagicMock(content="second"))],
        ),
    ]
    jpeg_b64 = base64.b64encode(b"\xff\xd8\xffcontent").decode()

    with patch.object(QwenAnnotator, "_encode_image", return_value=jpeg_b64):
        result = annotator.annotate(mock_image)

    assert result["result"] == ["first", "second"]
    assert result["model"] == "qwen-vl-max-latest"
    assert result["task"] == "vision_extraction"
    assert result["timestamp"] == 2
    assert result["image_path"] == mock_image
    assert openai.return_value.chat.completions.create.call_count == 2

    request = openai.return_value.chat.completions.create.call_args.kwargs
    assert request["max_tokens"] == 1200
    assert request["temperature"] == 0.2
    assert request["messages"][1]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
    }


def test_annotate_missing_image_returns_error():
    with (
        patch("openllm_ocr_annotator.annotators.qwen_annotator.OpenAI"),
        patch("openllm_ocr_annotator.annotators.qwen_annotator.PromptManager"),
        patch("openllm_ocr_annotator.utils.retry.time.sleep"),
    ):
        annotator = QwenAnnotator(api_key="test-key")
        result = annotator.annotate("/nonexistent/image.jpg")

    assert result["status"] == "error"
    assert "Image not found" in result["message"]


@pytest.mark.parametrize("annotator_type", ["qwen", "qwen2.5"])
def test_factory_creates_qwen_annotator(annotator_type):
    config = _make_config(type=annotator_type)

    with patch.object(QwenAnnotator, "from_config") as from_config:
        create_annotator(config)

    from_config.assert_called_once_with(config=config)
