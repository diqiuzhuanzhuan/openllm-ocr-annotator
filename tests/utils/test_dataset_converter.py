# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import pytest

from openllm_ocr_annotator.utils.dataset_converter import create_hf_dataset


def test_create_hf_dataset_encodes_fields_as_list_of_records(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.touch()
    result = {
        "result": {
            "fields": [
                {
                    "field_name": "invoice_number",
                    "value": "INV-001",
                    "confidence": 0.9,
                }
            ]
        },
        "metadata": {
            "image_path": str(image_path),
            "filename": "image.json",
            "task_id": "test_task",
            "timestamp": "2026-06-12 14:00:36",
            "annotators": ["annotator/model"],
        },
    }

    dataset = create_hf_dataset([result, result], {"train": 0.5, "test": 0.5})
    row = dataset["train"].select_columns(["fields", "metadata"])[0]

    assert row["fields"] == [
        {
            "field_name": "invoice_number",
            "value": "INV-001",
            "confidence": pytest.approx(0.9),
        }
    ]
    assert row["metadata"]["annotators"] == ["annotator/model"]
