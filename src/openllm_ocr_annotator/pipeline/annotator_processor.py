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


import logging
from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm
import multiprocessing as mp

logger = logging.getLogger(__name__)

class AnnotatorProcessor:
    """Processor for running single annotator on images."""
    
    def __init__(self, annotator, output_dir: Path):
        self.annotator = annotator
        self.output_dir = output_dir / annotator.__class__.__name__
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_image(self, image_path: str) -> Dict:
        """Process single image and save result."""
        img_path = Path(image_path)
        result_path = self.output_dir / f"{img_path.stem}.json"
        
        # Skip if already processed
        if result_path.exists():
            logger.info(f"Loading cached result for {img_path.name}")
            with open(result_path, 'r') as f:
                return json.load(f)
        
        try:
            # Run annotation
            result = self.annotator.annotate(str(img_path))
            
            # Save result
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None