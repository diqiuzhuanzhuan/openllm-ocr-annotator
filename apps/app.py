# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import argparse
from dotenv import load_dotenv
from src.openllm_ocr_annotator.config.config_manager import AnnotatorConfigManager
from src.openllm_ocr_annotator.pipeline import run_batch_annotation

argparser = argparse.ArgumentParser(description="OpenLLM OCR Annotator")

argparser.add_argument(
    "--config",
    type=str,
    default="examples/config.yaml",
    help="Path to the annotator config file.",
)

argparser.add_argument(
    "--max-workers",
    default=8,
    type=int,
    help="Maximum number of parallel workers for every annotator.",
)

if __name__ == "__main__":
    load_dotenv()
    args = argparser.parse_args()
    config_manager = AnnotatorConfigManager.from_file(args.config)

    annotator_configs = config_manager.get_enabled_annotators()
    weights = config_manager.get_annotator_weights()
    task_config = config_manager.get_task_config()
    ensemble_config = config_manager.get_ensemble_config()
    dataset_config = config_manager.get_dataset_config()

    run_batch_annotation(
        task_config=task_config,
        dataset_config=dataset_config,
        annotator_configs=annotator_configs,
        ensemble_config=ensemble_config,
        max_workers=args.max_workers,
        voting_weights=weights,
        max_files=task_config.max_files,
        create_dataset=dataset_config.enabled,
        dataset_split_ratio=dataset_config.split_ratio,
        num_samples=task_config.num_samples,
    )
