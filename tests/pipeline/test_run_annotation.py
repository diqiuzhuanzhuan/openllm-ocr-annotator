# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest.mock import patch

from openllm_ocr_annotator.config import (
    AnnotatorConfig,
    DatasetConfig,
    EnsembleConfig,
    TaskConfig,
)
from openllm_ocr_annotator.pipeline.run_annotation import run_batch_annotation


def test_dataset_output_dir_is_used_as_independent_root(tmp_path):
    dataset_root = tmp_path / "datasets"
    task = TaskConfig(
        task_id="task",
        input_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "annotations"),
    )
    dataset = DatasetConfig(output_dir=str(dataset_root), enabled=True)
    ensemble = EnsembleConfig(enabled=True)
    annotators = [AnnotatorConfig(model="openai/test")]
    image = tmp_path / "image.png"
    voted_dir = tmp_path / "voted"

    with (
        patch(
            "openllm_ocr_annotator.pipeline.run_annotation.collect_image_files",
            return_value=[image],
        ),
        patch("openllm_ocr_annotator.pipeline.run_annotation.run_parallel_annotation"),
        patch(
            "openllm_ocr_annotator.pipeline.run_annotation.run_voting_and_save",
            return_value=voted_dir,
        ),
        patch(
            "openllm_ocr_annotator.pipeline.run_annotation.convert_to_hf_if_needed"
        ) as convert,
    ):
        run_batch_annotation(task, dataset, annotators, ensemble)

    assert convert.call_args.args[-1] == Path(dataset_root)
