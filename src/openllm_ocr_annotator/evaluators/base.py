# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
import json
from utils.field_matcher import FieldMatcher, ExactMatcher
from typing import Optional


class BaseEvaluator(ABC):
    """Base class for evaluation metrics."""

    def __init__(
        self,
        ground_truth_dir: str,
        prediction_dir: str,
        field_matchers: Optional[Dict[str, FieldMatcher]] = None,
    ):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.prediction_dir = Path(prediction_dir)
        self.field_matchers: Dict[str, FieldMatcher] = {
            "default": ExactMatcher(),
        }
        if isinstance(field_matchers, dict) and all(
            [isinstance(matcher, FieldMatcher) for matcher in field_matchers.values()]
        ):
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
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def evaluate_single(self, ground_truth: Dict, prediction: Dict) -> Dict:
        """Evaluate single document."""
        pass

    @abstractmethod
    def evaluate_batch(self) -> Dict:
        """Evaluate all documents in directory."""
        pass
