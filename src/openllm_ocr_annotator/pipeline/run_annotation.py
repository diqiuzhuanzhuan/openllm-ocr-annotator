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


def prepare_output_dir(output_dir: str, task_id: str) -> Path:
    output_path = Path(output_dir) / task_id
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def collect_image_files(input_dir: str, max_files: int) -> List[Path]:
    image_files = get_image_files(input_dir)
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return []
    if max_files > 0:
        logger.info(f"Limiting to {max_files} files")
        image_files = image_files[:max_files]
    return image_files


def run_parallel_annotation(
    annotator_configs: List[AnnotatorConfig],
    output_path: Path,
    image_files: List[Path],
    max_workers: int = 8,
):
    processor = ParallelProcessor(
        annotator_configs=annotator_configs, output_dir=output_path, max_workers=max_workers
    )
    processor.run_parallel(image_files)


def run_voting_and_save(
    ensemble_config: EnsembleConfig,
    annotator_configs: List[AnnotatorConfig],
    output_path: Path,
    image_files: List[Path],
    format: str,
    task_id: str,
    num_samples:int = 1,
):
    ensemble_strategy = EnsembleStrategy.from_str(ensemble_config.method)
    if ensemble_strategy == EnsembleStrategy.SIMPLE_VOTE:
        voter = MajorityVoter()
    elif ensemble_strategy == EnsembleStrategy.WEIGHTED_VOTE:
        weights = {
            f"{ann_config.name}/{ann_config.model}": ann_config.weight
            for ann_config in annotator_configs
        }
        voter = WeightedVoter(weights=weights)
    elif ensemble_strategy == EnsembleStrategy.HIGHEST_CONFIDENCE:
        raise NotImplementedError("HighestConfidenceVoter is not implemented yet.")
    else:
        raise ValueError(f"Invalid voting strategy: {ensemble_strategy}")

    annotator_infos = [
        {"results_dir": output_path, "name": an_config.name, "model": an_config.model}
        for an_config in annotator_configs
    ]
    voting_manager = VotingManager(annotator_infos=annotator_infos, voter=voter)
    voted_dir = output_path / "voted_results"
    voted_dir.mkdir(exist_ok=True)

    for img_path in tqdm(image_files, desc="Computing voting results"):
        try:
            result = voting_manager.get_voted_result(image_path=img_path, output_dir=output_path,num_samples=num_samples)
            result["metadata"] = {
                "task_id": task_id,
                "image_path": str(img_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if "json" in format:
                save_as_json(result, voted_dir / f"{img_path.stem}.json")
            if "jsonl" in format:
                save_as_jsonl(result, voted_dir / f"{img_path.stem}.jsonl")
            if "tsv" in format:
                save_as_tsv(result, voted_dir / f"{img_path.stem}.tsv")
        except Exception as e:
            logger.error(f"Error in voting for {img_path}: {e}")
            continue
    return voted_dir


def convert_to_hf_if_needed(
    create_dataset, ensemble_config, dataset_split_ratio, voted_dir, output_path
):
    if not create_dataset:
        return
    if not ensemble_config or not ensemble_config.enabled:
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


def run_batch_annotation(
    input_dir: str,
    output_dir: str,
    annotator_configs: List[AnnotatorConfig],
    task_id: str,
    max_workers: int = 8,
    ensemble_config: EnsembleConfig | None = None,
    voting_weights: Optional[Dict[str, float]] = None,
    format: str | List[str] = "json",
    max_files: Optional[int] = -1,
    create_dataset: bool = True,
    dataset_split_ratio: Optional[Dict[str, float]] = None,
    num_samples: int = 1,
    **kwargs,
) -> None:
    """Run batch annotation with parallel processing."""
    try:
        output_path = prepare_output_dir(output_dir, task_id)
        image_files = collect_image_files(input_dir, max_files)
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return
        run_parallel_annotation(
            annotator_configs, output_path, image_files, max_workers
        )
        if ensemble_config and ensemble_config.enabled:
            voted_dir = run_voting_and_save(
                ensemble_config,
                annotator_configs,
                output_path,
                image_files,
                format,
                task_id,
                num_samples
            )
        if create_dataset and voted_dir:
            output_path = Path(task_config.output_dir) / Path(dataset_config.output_dir)
            convert_to_hf_if_needed(
                create_dataset,
                ensemble_config,
                dataset_split_ratio,
                voted_dir,
                output_path,
            )
            logger.info("Create dataset successfully.")
        logger.info(f"Completed task {task_id} with {len(image_files)} images")
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        raise e


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
        max_workers=task_config.max_workers,
        num_samples=task_config.num_samples,
    )
