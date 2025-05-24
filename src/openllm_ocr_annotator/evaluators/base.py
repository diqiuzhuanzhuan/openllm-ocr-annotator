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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
import json

class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, ground_truth_dir: str, prediction_dir: str):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.prediction_dir = Path(prediction_dir)
        
    def load_json(self, file_path: Path) -> Dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    @abstractmethod
    def evaluate_single(self, ground_truth: Dict, prediction: Dict) -> Dict:
        """Evaluate single document."""
        pass
    
    @abstractmethod
    def evaluate_batch(self) -> Dict:
        """Evaluate all documents in directory."""
        pass