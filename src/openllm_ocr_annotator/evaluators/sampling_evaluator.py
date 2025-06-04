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


from typing import Dict, List, Optional
from .base import BaseEvaluator
from .field_evaluator import FieldEvaluator
from collections import defaultdict
import json
from pathlib import Path
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SamplingEvaluator(BaseEvaluator):
    """Evaluator for assessing model performance with multiple sampling results."""

    def __init__(
        self,
        ground_truth_dir: str,
        predictions_base_dir: str,
        field_matchers: Optional[Dict] = None,
        num_samples: int = 128,
    ):
        """Initialize sampling evaluator.

        Args:
            ground_truth_dir: Directory containing ground truth files
            predictions_base_dir: Base directory containing sample directories
            field_matchers: Dictionary of field-specific matchers
            num_samples: Number of samples per image (default: 128)
        """
        super().__init__(ground_truth_dir, predictions_base_dir, field_matchers)
        self.num_samples = num_samples
        self.field_evaluator = FieldEvaluator(
            ground_truth_dir, predictions_base_dir, field_matchers
        )

    def _get_sample_predictions(self, image_id: str) -> List[Dict]:
        """Get all sample predictions for a single image.

        Args:
            image_id: Image identifier (filename without extension)

        Returns:
            List of prediction dictionaries from all samples
        """
        samples = []
        base_path = Path(self.prediction_dir)

        # Look for samples in numbered directories (sample_0, sample_1, etc.)
        for i in range(self.num_samples):
            sample_path = base_path / f"sample_{i}{image_id}.json"
            if sample_path.exists():
                try:
                    with open(sample_path) as f:
                        samples.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Error loading sample {i} for {image_id}: {e}")

        return samples

    def evaluate_single(self, ground_truth: Dict, image_id: str) -> Dict:
        """Evaluate single image across all its samples.

        Args:
            ground_truth: Ground truth annotation
            image_id: Image identifier

        Returns:
            Best scoring prediction and statistics
        """
        samples = self._get_sample_predictions(image_id)
        if not samples:
            return {
                "error": f"No samples found for {image_id}",
                "best_sample": None,
                "sample_stats": {},
            }

        # Evaluate each sample
        sample_scores = []
        for sample in samples:
            score = self.field_evaluator.evaluate_single(ground_truth, sample)
            sample_scores.append(
                {
                    "prediction": sample,
                    "accuracy": score["accuracy"],
                    "exact_match": score["exact_match"],
                    "field_results": score["field_results"],
                }
            )

        # Find best sample by accuracy
        best_sample = max(sample_scores, key=lambda x: x["accuracy"])

        # Calculate statistics
        accuracies = [s["accuracy"] for s in sample_scores]
        exact_matches = [s["exact_match"] for s in sample_scores]

        return {
            "best_sample": best_sample,
            "sample_stats": {
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "max_accuracy": max(accuracies),
                "min_accuracy": min(accuracies),
                "exact_match_rate": sum(exact_matches) / len(exact_matches),
                "total_samples": len(samples),
            },
        }

    def evaluate_batch(self) -> Dict:
        """Evaluate all images with their samples.

        Returns:
            Evaluation results including best samples and statistics
        """
        results = {
            "per_image": {},
            "overall_stats": defaultdict(float),
            "sampling_effectiveness": {},
        }

        total_images = 0

        for gt_file in self.ground_truth_dir.glob("*.json"):
            image_id = gt_file.stem
            try:
                ground_truth = self.load_json(gt_file)
                image_result = self.evaluate_single(ground_truth, image_id)
                results["per_image"][image_id] = image_result
                total_images += 1

                # Aggregate statistics
                if "best_sample" in image_result and image_result["best_sample"]:
                    stats = image_result["sample_stats"]
                    for key, value in stats.items():
                        results["overall_stats"][key] += value

            except Exception as e:
                logger.error(f"Error evaluating {image_id}: {e}")
                continue

        # Calculate averages
        if total_images > 0:
            for key in results["overall_stats"]:
                results["overall_stats"][key] /= total_images

        # Calculate sampling effectiveness
        accuracy_improvements = []
        for image_data in results["per_image"].values():
            if "sample_stats" in image_data:
                stats = image_data["sample_stats"]
                if "max_accuracy" in stats and "mean_accuracy" in stats:
                    improvement = stats["max_accuracy"] - stats["mean_accuracy"]
                    accuracy_improvements.append(improvement)

        if accuracy_improvements:
            results["sampling_effectiveness"] = {
                "mean_improvement": sum(accuracy_improvements)
                / len(accuracy_improvements),
                "max_improvement": max(accuracy_improvements),
                "min_improvement": min(accuracy_improvements),
            }

        return results

    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate detailed evaluation report."""
        report = []
        report.append("# Sampling Evaluation Report\n")

        # Overall metrics
        report.append("## Overall Statistics")
        overall = results["overall_stats"]
        report.append(f"- Mean Accuracy: {overall['mean_accuracy']:.2%}")
        report.append(f"- Best Sample Accuracy: {overall['max_accuracy']:.2%}")
        report.append(f"- Exact Match Rate: {overall['exact_match_rate']:.2%}")
        report.append(f"- Total Images Evaluated: {len(results['per_image'])}")
        report.append(f"- Samples per Image: {self.num_samples}\n")

        # Sampling effectiveness
        if "sampling_effectiveness" in results:
            eff = results["sampling_effectiveness"]
            report.append("## Sampling Effectiveness")
            report.append(f"- Mean Improvement: {eff['mean_improvement']:.2%}")
            report.append(f"- Max Improvement: {eff['max_improvement']:.2%}")
            report.append(f"- Min Improvement: {eff['min_improvement']:.2%}\n")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text
