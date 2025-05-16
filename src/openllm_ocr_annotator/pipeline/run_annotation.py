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

from ..annotators.openai import OpenAIAnnotator
from ....utils.formatter import save_as_jsonl, save_as_tsv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_batch_annotation(
    input_dir: str,
    output_dir: str,
    api_key: Optional[str] = None,
    task: str = "vision_extraction",
    variables: Optional[Dict[str, str]] = None,
    formats: Optional[list] = None
) -> None:
    """Run batch annotation on all images in input directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        api_key: Optional OpenAI API key
        task: Annotation task type
        variables: Optional template variables
        formats: List of output formats (default: ["jsonl", "tsv"])
    """
    try:
        # Initialize annotator
        annotator = OpenAIAnnotator(
            api_key=api_key,
            task=task
        )
        
        # Create output directories
        formats = formats or ["jsonl", "tsv"]
        output_paths = {}
        for fmt in formats:
            path = Path(output_dir) / fmt
            path.mkdir(parents=True, exist_ok=True)
            output_paths[fmt] = path
            
        # Get list of image files
        image_files = [
            f for f in Path(input_dir).iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return
            
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                logger.info(f"Processing {img_path.name}")
                
                # Get annotation result
                result = annotator.annotate(
                    str(img_path),
                    variables=variables
                )
                
                # Save in requested formats
                if "jsonl" in formats:
                    output_file = output_paths["jsonl"] / f"{img_path.stem}.jsonl"
                    save_as_jsonl(result, str(output_file))
                    
                if "tsv" in formats:
                    output_file = output_paths["tsv"] / f"{img_path.stem}.tsv"
                    save_as_tsv(result, str(output_file))
                    
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {str(e)}")
                continue
                
        logger.info(f"Completed processing {len(image_files)} images")
        
    except Exception as e:
        logger.error(f"Batch annotation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    run_batch_annotation(
        input_dir="data/images",
        output_dir="data/outputs",
        task="vision_extraction",
        variables={
            "document_type": "invoice",
            "fields_to_extract": """
                - Invoice Number
                - Total Amount
                - Date
                - Vendor Name
            """
        }
    )