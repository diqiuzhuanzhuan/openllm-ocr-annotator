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


from utils.logger import setup_logger
from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm
from src.openllm_ocr_annotator.annotators.base import BaseAnnotator
from utils.formatter import parse_json_from_text

logger = setup_logger(__name__)

class AnnotatorProcessor:
    """Processor for running single annotator on images."""
    
    def __init__(self, annotator: BaseAnnotator, output_dir: Path):
        """Initialize the processor with an annotator and output directory.
        
        Args:
            annotator (BaseAnnotator): The annotator to process images with
            output_dir (Path): Base output directory for results
        """
        self.annotator = annotator
        # Create output directory structure: {annotator_type}/{model_version}/
        model_version = getattr(annotator, "model", "default")
        name = getattr(annotator, "name", "openai")
        self.output_dir = output_dir / name / model_version
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self) -> Path:
        """Get the output directory for the current annotator."""
        return self.output_dir

    def process_images(self, image_files: List[Path]):
        """Process a list of images with the annotator."""
        """
        Args:
            image_files (List[Path]): List of image file paths to process. 
        """
        for img_path in tqdm(image_files, desc=f"Processing with {self.annotator.__class__.__name__}", unit="image"):
            self.process_single_image(str(img_path))
    
    def process_single_image(self, image_path: str) -> Dict:
        """Process single image and save result."""
        img_path = Path(image_path)
        result_path = self.output_dir / f"{img_path.stem}.json"
        
        # Skip if already processed
        if result_path.exists():
            #logger.info(f"Loading cached result for {img_path.name}")
            with open(result_path, 'r') as f:
                return json.load(f)
        
        try:
            # Run annotation and get result
            result = self.annotator.annotate(str(img_path))
            # Parse result if it's a string, otherwise use as is
            if isinstance(result, str):
                result = parse_json_from_text(result)
            elif isinstance(result, dict) and "result" in result:
                # If result is a dict with "result" key, parse that if it's a string
                if isinstance(result["result"], str):
                    result["result"] = parse_json_from_text(result["result"])
            
            if not result['result']:
                logger.warning(f"No valid annotation found for {img_path.name} while using {getattr(self.annotator, "model", "default")}")
                logger.warning(f"Result: {result}")
                return None
            
            # Save result
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None