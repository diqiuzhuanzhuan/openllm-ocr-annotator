# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from datasets import Dataset
from unittest.mock import MagicMock, patch

from openllm_ocr_annotator.config.config_manager import AnnotatorConfig
from openllm_ocr_annotator.pipeline.curator_processor import (
    CuratorAnnotatorProcessor,
    _merged_backend_params,
    _merged_generation_params,
)


def _config(**kwargs):
    defaults = dict(
        name="curator_test",
        type="curator",
        task="vision_extraction",
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=None,
        num_samples=1,
    )
    defaults.update(kwargs)
    return AnnotatorConfig(**defaults)


def test_save_response_dataset_uses_legacy_single_sample_layout(tmp_path):
    processor = CuratorAnnotatorProcessor.__new__(CuratorAnnotatorProcessor)
    processor.config = _config()
    processor.num_samples = 1
    processor.model_dir = tmp_path / "curator_test" / "gpt-4o-mini"

    dataset = Dataset.from_list(
        [
            {
                "stem": "a",
                "filename": "a.jpg",
                "image_path": str(tmp_path / "a.jpg"),
                "sample_id": 0,
                "result": '{"fields": []}',
            }
        ]
    )

    processor._save_response_dataset(dataset)

    result_path = processor.model_dir / "a.json"
    assert result_path.exists()
    assert '"fields": []' in result_path.read_text()


def test_run_calls_curator_llm_with_dataset_and_working_dir(tmp_path):
    img = tmp_path / "a.jpg"
    img.touch()
    config = _config(curator_working_dir=str(tmp_path / "curator-cache"))
    processor = CuratorAnnotatorProcessor(config, tmp_path)

    response = Dataset.from_list(
        [
            {
                "stem": "a",
                "filename": "a.jpg",
                "image_path": str(img),
                "sample_id": 0,
                "result": '{"fields": []}',
            }
        ]
    )
    llm = MagicMock(return_value=response)

    with patch.object(processor, "_create_llm", return_value=llm):
        processor.run([img])

    called_dataset = llm.call_args.args[0]
    assert len(called_dataset) == 1
    assert llm.call_args.kwargs["working_dir"] == str(tmp_path / "curator-cache")
    assert (tmp_path / "curator_test" / "gpt-4o-mini" / "a.json").exists()


def test_task_num_samples_overrides_annotator_default(tmp_path):
    processor = CuratorAnnotatorProcessor(
        _config(num_samples=1), tmp_path, num_samples=2
    )

    assert processor.num_samples == 2


def test_curator_params_prefer_nested_provider_config():
    config = _config(
        model="gpt-4o",
        max_tokens=1000,
        tpm=100,
        backend_params={
            "require_all_responses": False,
            "max_tokens_per_minute": 8_000_000,
        },
        generation_params={"max_tokens": 4096},
    )

    backend_params = _merged_backend_params(config)
    generation_params = _merged_generation_params(config)

    assert backend_params["require_all_responses"] is False
    assert backend_params["max_tokens_per_minute"] == 8_000_000
    assert generation_params["max_tokens"] == 4096
