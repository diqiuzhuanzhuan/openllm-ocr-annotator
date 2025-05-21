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
from src.openllm_ocr_annotator.annotators.openai_annotator import OpenAIAnnotator
from src.openllm_ocr_annotator.annotators.claude_annotator import ClaudeAnnotator
from src.openllm_ocr_annotator.annotators.gemini_annotator import GeminiAnnotator
from src.openllm_ocr_annotator.pipeline.parallel_processor import ParallelProcessor
from utils.formatter import save_as_json, save_as_jsonl, save_as_tsv
from utils.file_utils import get_image_files
from utils.dataset_converter import convert_to_hf_dataset
from utils.logger import setup_logger

logging.basicConfig(level=logging.INFO)
logger = setup_logger(__name__)


def run_batch_annotation(
    input_dir: str,
    output_dir: str,
    annotator_configs: List[Dict],
    task_id: str,  # 添加任务标识符
    voting_strategy: str = "majority",
    voting_weights: Optional[Dict[str, float]] = None,
    format: str | List[str] = "json",
    max_files: Optional[int] = -1,
    create_dataset: bool = True,
    dataset_split_ratio: Optional[Dict[str, float]] = None,
    **kwargs
) -> None:
    """Run batch annotation with parallel processing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        annotator_configs: List of annotator configurations
        task_id: Unique identifier for the annotation task
        voting_strategy: Strategy for voting ("majority" or "weighted")
        voting_weights: Optional weights for weighted voting strategy.
                      Example: {
                          "OpenAIAnnotator/gpt-4-vision-preview": 1.0,
                          "OpenAIAnnotator/gpt-3.5-turbo-vision": 0.8,
                          "ClaudeAnnotator/claude-3-opus": 0.9
                      }
        format: Output format(s)
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
        
        # Run parallel annotation
        processor = ParallelProcessor(annotators, output_path)
        processor.run_parallel(image_files)
        
        # After all annotations are complete, run voting
        if voting_strategy == "majority":
            voter = MajorityVoter()
        elif voting_strategy == "weighted":
            voter = WeightedVoter(weights=voting_weights)
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")
            
        annotator_paths = [
            {
                "results_dir": output_path,
                "name": "OpenAIAnnotator",
                "model": an_config["model"] 
            }
         for an_config in annotator_configs]
            
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
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
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
            try:
                dataset_split_ratio = dataset_split_ratio or {
                    "train": 0.8,
                    "test": 0.1,
                    "validation": 0.1
                }
                dataset_dir = output_path / "dataset"
                logger.info("Converting results to HuggingFace dataset format...")
                dataset = convert_to_hf_dataset(
                    voted_dir=str(voted_dir),
                    output_dir=str(dataset_dir),
                    split_ratio=dataset_split_ratio
                )
                logger.info(f"Dataset created and saved to {dataset_dir}")
            except Exception as e:
                logger.error(f"Failed to create dataset: {e}")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with different model versions and weights
    annotator_configs = [
        # GPT-4 Vision Preview - Most accurate but expensive
        {
            "type": "openai",
            "api_key": "sk-xxx",
            "model": "gpt-4-vision-preview"
        },
        # GPT-3.5 Turbo Vision - Fast but less accurate
        {
            "type": "openai",
            "api_key": "sk-xxx",
            "model": "gpt-3.5-turbo-vision"
        },
        # Claude 3 Opus - Strong capabilities
        {
            "type": "claude",
            "api_key": "sk-yyy",
            "model": "claude-3-opus-20240229"
        },
        # Claude 3 Sonnet - Good balance
        {
            "type": "claude",
            "api_key": "sk-yyy",
            "model": "claude-3-sonnet-20240229"
        },
        # Gemini Pro Vision - Reliable baseline
        {
            "type": "gemini",
            "api_key": "sk-zzz",
            "model": "gemini-pro-vision"
        }
    ]

    # Configure weights based on model capabilities
    weights = {
        "OpenAIAnnotator/gpt-4-vision-preview": 1.0,    # Highest weight for best accuracy
        "OpenAIAnnotator/gpt-3.5-turbo-vision": 0.7,    # Lower weight due to limitations
        "ClaudeAnnotator/claude-3-opus-20240229": 0.95, # Very high confidence
        "ClaudeAnnotator/claude-3-sonnet-20240229": 0.8,# Good balance
        "GeminiAnnotator/gemini-pro-vision": 0.75       # Reliable baseline
    }

    # Run batch annotation with weighted voting
    run_batch_annotation(
        input_dir="data/images",
        output_dir="data/outputs",
        task_id="foreign_trade_20250519",
        annotator_configs=annotator_configs,
        voting_strategy="weighted",
        voting_weights=weights
    )