# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Inject a stub 'litellm' module before the annotator module is imported so
# that the lazy `import litellm` inside annotate() picks up our mock.
# ---------------------------------------------------------------------------
_litellm_stub = MagicMock()
sys.modules.setdefault("litellm", _litellm_stub)

from openllm_ocr_annotator.config.config_manager import AnnotatorConfig  # noqa: E402
from src.openllm_ocr_annotator.annotators.litellm_annotator import LiteLLMAnnotator  # noqa: E402


def _make_config(**kwargs):
    defaults = dict(
        name="litellm_test",
        type="litellm",
        task="vision_extraction",
        api_key="test-api-key",
        model="anthropic/claude-3-opus-20240229",
        num_samples=1,
        temperature=None,
    )
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


def _make_mock_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestFromConfig:
    def test_from_config_creates_annotator(self):
        config = _make_config()
        with patch.object(LiteLLMAnnotator, "__init__", return_value=None) as mock_init:
            LiteLLMAnnotator.from_config(config)
            mock_init.assert_called_once_with(
                name="litellm_test",
                api_key="test-api-key",
                model="anthropic/claude-3-opus-20240229",
                task="vision_extraction",
                max_tokens=config.max_tokens,
                temperature=None,
                base_url=config.base_url,
                prompt_path=config.prompt_path,
                n=1,
            )


class TestGetProvider:
    def test_get_provider_with_prefix(self):
        annotator = LiteLLMAnnotator.__new__(LiteLLMAnnotator)
        annotator.model = "anthropic/claude-3-opus-20240229"
        assert annotator._get_provider() == "anthropic"

    def test_get_provider_mistral(self):
        annotator = LiteLLMAnnotator.__new__(LiteLLMAnnotator)
        annotator.model = "mistral/mistral-large-latest"
        assert annotator._get_provider() == "mistral"

    def test_get_provider_without_prefix(self):
        annotator = LiteLLMAnnotator.__new__(LiteLLMAnnotator)
        annotator.model = "gpt-4-vision-preview"
        assert annotator._get_provider() == "openai"

    def test_get_prompt_type_matches_provider(self):
        annotator = LiteLLMAnnotator.__new__(LiteLLMAnnotator)
        annotator.model = "groq/llama-3.2-90b-vision-preview"
        assert annotator._get_prompt_type() == annotator._get_provider()


class TestAnnotate:
    @pytest.fixture
    def mock_image(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        return str(img)

    @pytest.fixture
    def annotator(self):
        config = _make_config()
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            return LiteLLMAnnotator.from_config(config)

    def test_annotate_returns_result_list(self, annotator, mock_image):
        annotator.prompt_manager = MagicMock(
            get_prompt=MagicMock(return_value={"system": "sys", "user": "user"})
        )
        _litellm_stub.completion.return_value = _make_mock_response('{"fields": []}')

        with patch.object(LiteLLMAnnotator, "_encode_image", return_value="base64data"):
            result = annotator.annotate(mock_image)

        assert "result" in result
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 1
        assert result["model"] == "anthropic/claude-3-opus-20240229"
        assert result["task"] == "vision_extraction"
        assert result["image_path"] == mock_image

    def test_annotate_multiple_samples(self, mock_image):
        config = _make_config(num_samples=3)
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            annotator = LiteLLMAnnotator.from_config(config)

        annotator.prompt_manager = MagicMock(
            get_prompt=MagicMock(return_value={"system": "sys", "user": "user"})
        )
        _litellm_stub.completion.return_value = _make_mock_response("response text")
        _litellm_stub.completion.reset_mock()

        with patch.object(LiteLLMAnnotator, "_encode_image", return_value="base64data"):
            result = annotator.annotate(mock_image)

        assert len(result["result"]) == 3
        assert _litellm_stub.completion.call_count == 3

    def test_annotate_image_not_found_raises(self, annotator):
        result = annotator.annotate("/nonexistent/path/image.jpg")
        # retry_with_backoff returns {"status": "error", ...} after exhausting retries
        assert result.get("status") == "error"


class TestApiKeyEnvVar:
    def test_api_key_sets_env_var_for_known_provider(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            LiteLLMAnnotator(
                api_key="my-secret-key",
                model="anthropic/claude-3-opus-20240229",
            )
        assert os.environ.get("ANTHROPIC_API_KEY") == "my-secret-key"

    def test_api_key_sets_groq_env_var(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            LiteLLMAnnotator(
                api_key="groq-key",
                model="groq/llama-3.2-90b-vision-preview",
            )
        assert os.environ.get("GROQ_API_KEY") == "groq-key"


class TestApiUrlEnvVar:
    def test_provider_api_url(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/v1")
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            annotator = LiteLLMAnnotator(
                model="anthropic/claude-3-opus-20240229",
            )
        assert annotator.base_url == "https://anthropic.example/v1"

    def test_explicit_url_has_priority(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example/v1")
        with patch(
            "src.openllm_ocr_annotator.annotators.litellm_annotator.PromptManager"
        ):
            annotator = LiteLLMAnnotator(
                model="anthropic/claude-3-opus-20240229",
                base_url="https://config.example/v1",
            )
        assert annotator.base_url == "https://config.example/v1"
