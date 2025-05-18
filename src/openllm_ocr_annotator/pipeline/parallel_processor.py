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
import multiprocessing as mp
from typing import List, Dict
from .annotator_processor import AnnotatorProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Manages parallel processing of multiple annotators."""
    
    def __init__(self, annotators: List, output_dir: Path):
        self.processors = [
            AnnotatorProcessor(annotator, output_dir)
            for annotator in annotators
        ]
        self.output_dir = output_dir
    
    def run_annotator_process(self, processor: AnnotatorProcessor, image_files: List[Path]):
        """Run single annotator process."""
        for img_path in tqdm(image_files, desc=f"Processing with {processor.annotator_name}", unit="image"):
            processor.process_single_image(str(img_path))
    
    def run_parallel(self, image_files: List[Path]):
        """Run all annotators in parallel."""
        processes = []
        
        for processor in self.processors:
            p = mp.Process(
                target=self.run_annotator_process,
                args=(processor, image_files)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()

