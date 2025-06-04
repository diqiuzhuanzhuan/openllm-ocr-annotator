# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import logging
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
from typing import List
from src.openllm_ocr_annotator.voters.majority import MajorityVoter
from src.openllm_ocr_annotator.voters.weighted import WeightedVoter
from src.openllm_ocr_annotator.voters.manager import VotingManager
from src.openllm_ocr_annotator.pipeline.parallel_processor import ParallelProcessor
from src.openllm_ocr_annotator.config import AnnotatorConfig
from src.openllm_ocr_annotator.config import EnsembleStrategy, EnsembleConfig
from utils.formatter import save_as_json, save_as_jsonl, save_as_tsv
from utils.file_utils import get_image_files
from utils.dataset_converter import convert_to_hf_dataset
from utils.logger import setup_logger


logging.basicConfig(level=logging.INFO)
logger = setup_logger(__name__)


def run_batch_annotation(
    input_dir: str,
    output_dir: str,
    annotator_configs: List[AnnotatorConfig],
    task_id: str,  # 添加任务标识符
    max_workers: int = 8,
    ensemble_config: EnsembleConfig | None = None,
    voting_weights: Optional[Dict[str, float]] = None,
    format: str | List[str] = "json",
    max_files: Optional[int] = -1,
    create_dataset: bool = True,
    dataset_split_ratio: Optional[Dict[str, float]] = None,
    **kwargs,
) -> None:
    """Run batch annotation with parallel processing.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        annotator_configs: List of annotator configurations
        task_id: Unique identifier for the annotation task
        max_workers: Maximum number of parallel workers for every annotator
        voting_strategy: Strategy for voting (VotingStrategy.MAJORITY or VotingStrategy.WEIGHTED)
        voting_weights: Optional weights for weighted voting strategy.
                      Example: {
                          "OpenAIAnnotator/gpt-4-vision-preview": 1.0,
                          "OpenAIAnnotator/gpt-3.5-turbo-vision": 0.8,
                          "ClaudeAnnotator/claude-3-opus": 0.9
                      }
        format: Output format(s)
        max_files: the maximum number of processed files, -1 means all
        create_dataset: True for creating huggingface format dataset
        dataset_split_ratio: the ratio for splitting the dataset, e.g. {"train": 0.8, "validation": 0.1, "test": 0.1}
                             If a single float is provided, it will be used as the train split and the rest as test.
    """
    try:
        # Create output directory with task_id
        output_path = Path(output_dir) / task_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = get_image_files(input_dir)
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return

        # Limit number of files if max_files is specified
        if max_files > 0:
            logger.info(f"Limiting to {max_files} files")
            image_files = image_files[:max_files]
        # Run parallel annotation
        processor = ParallelProcessor(
            annotator_configs, output_path, max_workers=max_workers
        )
        processor.run_parallel(image_files)
        if ensemble_config and ensemble_config.enabled:
            # Convert string to enum if needed
            ensemble_strategy = EnsembleStrategy.from_str(ensemble_config.method)

            # After all annotations are complete, run voting
            if ensemble_strategy == EnsembleStrategy.SIMPLE_VOTE:
                voter = MajorityVoter()
            elif ensemble_strategy == EnsembleStrategy.WEIGHTED_VOTE:
                voter = WeightedVoter(weights=voting_weights)
            elif ensemble_strategy == EnsembleStrategy.HIGHEST_CONFIDENCE:
                # TODO: implement HighestConfidenceVoter
                # voter = HighestConfidenceVoter()
                raise NotImplementedError(
                    "HighestConfidenceVoter is not implemented yet."
                )
            else:
                raise ValueError(f"Invalid voting strategy: {ensemble_strategy}")

            annotator_paths = [
                {
                    "results_dir": output_path,
                    "name": an_config.name,
                    "model": an_config.model,
                }
                for an_config in annotator_configs
            ]

            voting_manager = VotingManager(annotator_paths=annotator_paths, voter=voter)

            # Process voting for each image
            voted_dir = output_path / "voted_results"
            voted_dir.mkdir(exist_ok=True)

            for img_path in tqdm(image_files, desc="Computing voting results"):
                try:
                    # Get voted result
                    result = voting_manager.get_voted_result(img_path, output_path)

                    # Add task metadata
                    result["metadata"] = {
                        "task_id": task_id,
                        "image_path": str(img_path),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Save voted result in requested formats
                    if "json" in format:
                        save_as_json(result, voted_dir / f"{img_path.stem}.json")
                    if "jsonl" in format:
                        save_as_jsonl(result, voted_dir / f"{img_path.stem}.jsonl")
                    if "tsv" in format:
                        save_as_tsv(result, voted_dir / f"{img_path.stem}.tsv")

                except Exception as e:
                    logger.error(f"Error in voting for {img_path}: {e}")
                    continue

        logger.info(f"Completed task {task_id} with {len(image_files)} images")

        # Convert to HuggingFace dataset if requested
        if create_dataset:
            if not ensemble_config.enabled:
                logger.warning("Dataset creation is only supported in ensemble mode.")
                return
            try:
                if isinstance(dataset_split_ratio, float):
                    dataset_split_ratio = {
                        "train": dataset_split_ratio,
                        "test": 1 - dataset_split_ratio,
                    }
                else:
                    dataset_split_ratio = dataset_split_ratio or {
                        "train": 0.8,
                        "test": 0.1,
                        "validation": 0.1,
                    }
                dataset_dir = output_path / "dataset"
                logger.info("Converting results to HuggingFace dataset format...")
                convert_to_hf_dataset(
                    voted_dir=str(voted_dir),
                    output_dir=str(dataset_dir),
                    split_ratio=dataset_split_ratio,
                )
                logger.info(f"Dataset created and saved to {dataset_dir}")
            except Exception as e:
                logger.error(f"Failed to create dataset: {e}")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        raise


if __name__ == "__main__":
    from src.openllm_ocr_annotator.config.config_manager import AnnotatorConfigManager

    config_manager = AnnotatorConfigManager.from_file("examples/config.yaml")
    annotator_configs = config_manager.get_enabled_annotators()
    weights = config_manager.get_annotator_weights()
    dataset_config = config_manager.get_dataset_config()
    ensemble_config = config_manager.get_ensemble_config()

    # Run batch annotation with weighted voting and create dataset
    task_config = config_manager.get_task_config()
    run_batch_annotation(
        input_dir=task_config.input_dir,
        output_dir=task_config.output_dir,
        task_id=task_config.task_id,
        annotator_configs=annotator_configs,
        ensemble_strategy=ensemble_config.method,
        voting_weights=weights,
        max_files=task_config.max_files,
        create_dataset=dataset_config.enabled,
        dataset_split_ratio=dataset_config.split_ratio,
    )
