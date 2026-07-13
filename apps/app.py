# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from openllm_ocr_annotator.config.config_manager import (
    AnnotatorConfigManager,
    register_config_store,
)
from openllm_ocr_annotator.pipeline import run_batch_annotation


register_config_store()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    load_dotenv()
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    config_manager = AnnotatorConfigManager.from_omegaconf(config)

    annotator_configs = config_manager.get_enabled_annotators()
    task_config = config_manager.get_task_config()
    ensemble_config = config_manager.get_ensemble_config()
    dataset_config = config_manager.get_dataset_config()

    run_batch_annotation(
        task_config=task_config,
        dataset_config=dataset_config,
        annotator_configs=annotator_configs,
        ensemble_config=ensemble_config,
        format=ensemble_config.output_format,
        max_files=task_config.max_files,
        create_dataset=dataset_config.enabled,
        dataset_split_ratio=dataset_config.split_ratio,
        num_samples=task_config.num_samples,
    )


if __name__ == "__main__":
    main()
