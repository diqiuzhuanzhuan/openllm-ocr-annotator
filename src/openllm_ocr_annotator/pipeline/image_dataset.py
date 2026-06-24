# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Iterable

from datasets import Dataset

from openllm_ocr_annotator.config import AnnotatorConfig


def model_output_dir(output_dir: Path, config: AnnotatorConfig) -> Path:
    """Return the legacy-compatible output directory for one annotator."""
    return output_dir / config.name.lstrip("/") / (config.model or "default").lstrip("/")


def result_path_for_row(base_dir: Path, stem: str, sample_id: int | None = None) -> Path:
    """Return where a row should be saved in the legacy JSON layout."""
    if sample_id is None:
        return base_dir / f"{stem}.json"
    return base_dir / "sampling" / f"sample_{sample_id}" / f"{stem}.json"


def build_image_dataset(
    image_files: Iterable[Path],
    *,
    num_samples: int = 1,
    output_dir: Path | None = None,
) -> Dataset:
    """Build a curator input dataset from image paths.

    When output_dir is provided, rows with existing output JSON files are omitted.
    """
    rows = []
    sample_count = max(num_samples or 1, 1)
    for image_path in image_files:
        for sample_id in range(sample_count):
            effective_sample_id = sample_id if sample_count > 1 else None
            if output_dir is not None:
                result_path = result_path_for_row(
                    output_dir, image_path.stem, effective_sample_id
                )
                if result_path.exists():
                    continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "filename": image_path.name,
                    "stem": image_path.stem,
                    "sample_id": sample_id,
                    "variables": "{}",
                }
            )
    return Dataset.from_list(rows)
