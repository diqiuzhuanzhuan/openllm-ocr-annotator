# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import base64
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openllm_ocr_annotator.annotators.claude_annotator import ClaudeAnnotator
from openllm_ocr_annotator.config import AnnotatorConfig
from openllm_ocr_annotator.pipeline.annotator_processor import create_annotator


def _make_config(**kwargs):
    defaults = {
        "name": "claude_test",
        "type": "claude",
        "task": "vision_extraction",
        "api_key": "test-api-key",
        "model": "claude-sonnet-4-6",
        "max_tokens": 1200,
        "temperature": None,
        "num_samples": 1,
    }
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


@pytest.fixture
def anthropic_stub(monkeypatch):
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    module = SimpleNamespace(Anthropic=constructor)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return constructor, client


@pytest.fixture
def mock_image(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    return str(image)


def test_from_config_passes_all_supported_options():
    config = _make_config(num_samples=2, base_url="https://anthropic.example/v1")

    with patch.object(ClaudeAnnotator, "__init__", return_value=None) as init:
        ClaudeAnnotator.from_config(config)

    init.assert_called_once_with(
        name="claude_test",
        api_key="test-api-key",
        model="claude-sonnet-4-6",
        task="vision_extraction",
        max_tokens=1200,
        temperature=None,
        base_url="https://anthropic.example/v1",
        prompt_path=None,
        n=2,
    )


def test_init_uses_environment_and_custom_base_url(monkeypatch, anthropic_stub):
    constructor, _ = anthropic_stub
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/v1")

    with patch("openllm_ocr_annotator.annotators.claude_annotator.PromptManager"):
        annotator = ClaudeAnnotator()

    constructor.assert_called_once_with(
        api_key="environment-key",
        base_url="https://anthropic.example/v1",
    )
    assert annotator.base_url == "https://anthropic.example/v1"


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeAnnotator()


def test_media_type_detects_png():
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\ncontent").decode()

    assert ClaudeAnnotator._media_type(encoded) == "image/png"


def test_response_text_joins_all_text_blocks():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="first"),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(type="text", text=" second"),
        ]
    )

    assert ClaudeAnnotator._response_text(response) == "first second"


def test_annotate_uses_messages_api_and_returns_samples(mock_image, anthropic_stub):
    _, client = anthropic_stub
    with patch("openllm_ocr_annotator.annotators.claude_annotator.PromptManager"):
        annotator = ClaudeAnnotator(
            api_key="test-key",
            model="claude-sonnet-4-6",
            max_tokens=1200,
            temperature=0.2,
            n=2,
        )

    annotator.prompt_manager.get_prompt.return_value = {
        "system": "system prompt",
        "user": "user prompt",
    }
    client.messages.create.side_effect = [
        SimpleNamespace(content=[SimpleNamespace(type="text", text="first")]),
        SimpleNamespace(content=[SimpleNamespace(type="text", text="second")]),
    ]
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\ncontent").decode()

    with patch.object(ClaudeAnnotator, "_encode_image", return_value=png_b64):
        result = annotator.annotate(mock_image)

    assert result["result"] == ["first", "second"]
    assert result["model"] == "claude-sonnet-4-6"
    assert result["task"] == "vision_extraction"
    assert result["image_path"] == mock_image
    assert client.messages.create.call_count == 2

    request = client.messages.create.call_args.kwargs
    assert request["max_tokens"] == 1200
    assert request["temperature"] == 0.2
    assert request["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_annotate_missing_image_returns_error(anthropic_stub):
    with (
        patch("openllm_ocr_annotator.annotators.claude_annotator.PromptManager"),
        patch("openllm_ocr_annotator.utils.retry.time.sleep"),
    ):
        annotator = ClaudeAnnotator(api_key="test-key")
        result = annotator.annotate("/nonexistent/image.jpg")

    assert result["status"] == "error"
    assert "Image not found" in result["message"]


def test_factory_creates_claude_annotator():
    config = _make_config()

    with patch.object(ClaudeAnnotator, "from_config") as from_config:
        create_annotator(config)

    from_config.assert_called_once_with(config=config)
