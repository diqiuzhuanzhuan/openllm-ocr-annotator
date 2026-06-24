# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from openllm_ocr_annotator.pipeline.image_dataset import build_image_dataset


def test_build_image_dataset_repeats_rows_for_samples(tmp_path):
    img = tmp_path / "a.jpg"
    img.touch()

    dataset = build_image_dataset([img], num_samples=2)

    assert len(dataset) == 2
    assert dataset[0]["image_path"] == str(img)
    assert dataset[0]["sample_id"] == 0
    assert dataset[1]["sample_id"] == 1


def test_build_image_dataset_skips_existing_results(tmp_path):
    img = tmp_path / "a.jpg"
    img.touch()
    output_dir = tmp_path / "out"
    sample_dir = output_dir / "sampling" / "sample_0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "a.json").write_text("{}")

    dataset = build_image_dataset([img], num_samples=2, output_dir=output_dir)

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == 1
