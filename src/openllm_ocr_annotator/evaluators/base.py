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
from utils.field_matcher import FieldMatcher, ExactMatcher
from typing import Optional

class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, ground_truth_dir: str, prediction_dir: str,
        field_matchers: Optional[Dict[str, FieldMatcher]] = None):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.prediction_dir = Path(prediction_dir)
        self.field_matchers: Dict[str, FieldMatcher] = {
            "default": ExactMatcher(),
        }
        if isinstance(field_matchers, dict)  and all([isinstance(matcher, FieldMatcher) for matcher in field_matchers.values()]):
            self.field_matchers.update(field_matchers)
    
    def get_matcher(self, field_name: str) -> FieldMatcher:
        """Get the appropriate matcher for a given field.

        Args:
            field_name: Name of the field

        Returns:
            FieldMatcher: Matching strategy for the field
        """
        return self.field_matchers.get(field_name, self.field_matchers["default"])

        
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