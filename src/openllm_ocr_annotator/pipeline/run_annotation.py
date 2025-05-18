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

import os
import logging
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
from typing import List
from src.openllm_ocr_annotator.voters.majority import MajorityVoter
from src.openllm_ocr_annotator.voters.manager import VotingManager
from openllm_ocr_annotator.annotators.openai_annotator import OpenAIAnnotator
from src.openllm_ocr_annotator.annotators.claude_annotator import ClaudeAnnotator
from src.openllm_ocr_annotator.annotators.gemini_annotator import GeminiAnnotator
from utils.formatter import save_as_jsonl, save_as_tsv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_batch_annotation(
    input_dir: str,
    output_dir: str,
    annotator_configs: List[Dict],
    voting_strategy: str = "majority",
    **kwargs
) -> None:
    """Run batch annotation with separate annotation and voting phases."""
    
    try:
        # Initialize annotators
        annotators = []
        for config in annotator_configs:
            if config["type"] == "openai":
                annotators.append(OpenAIAnnotator(config["api_key"], model=config.get("model")))
            elif config["type"] == "claude":
                annotators.append(ClaudeAnnotator(config["api_key"], model=config.get("model")))
            elif config["type"] == "gemini":
                annotators.append(GeminiAnnotator(config["api_key"], model=config.get("model")))
                
        # Initialize voter and manager
        voter = MajorityVoter()
        voting_manager = VotingManager(annotators, voter)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process images
        image_files = get_image_files(input_dir)

        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                logger.info(f"\nProcessing {img_path.name}")
                
                # Get or compute voted result
                result = voting_manager.get_voted_result(
                    str(img_path),
                    output_path
                )
                
                logger.info(f"Successfully processed {img_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
                
        logger.info(f"Completed processing {len(image_files)} images")
        
    except Exception as e:
        logger.error(f"Batch annotation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    annotator_configs = [
    {
        "type": "openai",
        "api_key": "sk-xxx",
        "model": "gpt-4-vision-preview"
    },
    {
        "type": "claude",
        "api_key": "sk-yyy",
        "model": "claude-3-opus-20240229"
    },
    {
        "type": "gemini",
        "api_key": "sk-zzz",
        "model": "gemini-pro-vision"
    }
    ]

    run_batch_annotation(
        input_dir="data/images",
        output_dir="data/outputs",
        annotator_configs=annotator_configs,
        voting_strategy="majority"
    )