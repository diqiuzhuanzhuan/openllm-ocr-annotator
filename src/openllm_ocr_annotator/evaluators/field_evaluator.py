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


from typing import Dict, Optional, List, Tuple
from .base import BaseEvaluator
from utils.logger import setup_logger
from utils.field_matcher import NumericMatcher, DateMatcher, ExactMatcher, FieldMatcher, CaseInsensitiveMatcher
from collections import defaultdict


logger = setup_logger(__name__)

class FieldEvaluator(BaseEvaluator):

    def evaluate_single(self, ground_truth: Dict, prediction: Dict) -> Dict:
        """Evaluate single document field accuracy.
        
        Args:
            ground_truth: Ground truth annotation
            prediction: Predicted annotation
            
        Returns:
            Dictionary containing:
            - field_results: Per-field accuracy
            - exact_match: Whether all fields match
            - accuracy: Percentage of correct fields
        """
        gt_fields = {f["field_name"]: f["value"] for f in ground_truth.get('result', {}).get("fields", [])}
        pred_fields = {f["field_name"]: f["value"] for f in prediction.get('result', {}).get("fields", [])}
        
        results = {
            "field_results": {},
            "exact_match": False,
            "accuracy": 0.0,
            "field_count": len(gt_fields)
        }
        
        if not gt_fields:
            logger.warning("Ground truth contains no fields")
            return results
            
        correct_count = 0
        for field_name, gt_value in gt_fields.items():
            pred_value = pred_fields.get(field_name)
            matcher = self.get_matcher(field_name)
            is_correct = matcher.match(gt_value, pred_value) if pred_value is not None else False
            correct_count += int(is_correct)
            
            results["field_results"][field_name] = {
                "correct": is_correct,
                "ground_truth": gt_value,
                "prediction": pred_value
            }
            
        results["accuracy"] = correct_count / len(gt_fields)
        results["exact_match"] = (correct_count == len(gt_fields))
        
        return results

    def evaluate_batch(self) -> Dict:
        """Evaluate all documents in directory.
    
        Returns:
            Dictionary containing:
            - per_document: Results for each document
            - field_accuracy: Accuracy per field type
            - overall_accuracy: Average document accuracy 
            - document_perfect_match_rate: Percentage of documents where all fields match
            - exact_match_count: Number of perfectly matched documents
            - total_documents: Total number of evaluated documents
        """
        results = {
            "per_document": {},
            "field_accuracy": defaultdict(lambda: {"correct": 0, "total": 0}),
            "overall_accuracy": 0.0,
            "document_perfect_match_rate": 0.0,
            "exact_match_count": 0,
            "total_documents": 0
        }
        
        total_docs = 0
        exact_matches = 0
        
        for gt_file in self.ground_truth_dir.glob("*.json"):
            doc_id = gt_file.stem
            pred_file = self.prediction_dir / gt_file.name
            total_docs += 1
            
            if not pred_file.exists():
                logger.warning(f"No prediction found for {doc_id}")
                continue
                
            try:
                gt_data = self.load_json(gt_file)
                pred_data = self.load_json(pred_file)
                doc_result = self.evaluate_single(gt_data, pred_data)
                
                results["per_document"][doc_id] = doc_result
                
                # Update field-level statistics
                for field_name, field_result in doc_result["field_results"].items():
                    results["field_accuracy"][field_name]["total"] += 1
                    if field_result["correct"]:
                        results["field_accuracy"][field_name]["correct"] += 1
                
                # Track perfect matches (all fields correct)
                if doc_result["exact_match"]:
                    exact_matches += 1
                
            except Exception as e:
                logger.warning(f"Error evaluating {doc_id}: {e}")
                continue
        
        # Calculate final metrics
        if total_docs > 0:
            results["document_perfect_match_rate"] = exact_matches / total_docs
            results["exact_match_count"] = exact_matches
            results["total_documents"] = total_docs
            results["overall_accuracy"] = sum(
                doc["accuracy"] for doc in results["per_document"].values()
            ) / total_docs
            
            # Convert field accuracy to percentages
            results["field_accuracy"] = {
                field: {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct": stats["correct"],
                    "total": stats["total"]
                }
                for field, stats in results["field_accuracy"].items()
            }
            
        return results


    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate evaluation report in markdown format."""
        report = []
        report.append("# Evaluation Report\n")
        
        # Overall metrics
        report.append("## Overall Metrics")
        report.append(f"- Total documents evaluated: {results['total_documents']}")
        report.append(f"- Documents with perfect match: {results['exact_match_count']}")
        report.append(f"- Document perfect match rate: {results['document_perfect_match_rate']:.2%}")
        report.append(f"- Average field accuracy: {results['overall_accuracy']:.2%}\n")
        
        # Field-level accuracy
        report.append("####             Field-level Accuracy               ####")
        report.append("|          Field          | Accuracy | Correct | Total  |")
        report.append("|-------------------------|----------|---------|--------|")
        for field, stats in results["field_accuracy"].items():
            report.append(
                f"| {field} | {stats['accuracy']:.2%} | {stats['correct']} | {stats['total']} |"
            )
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
                
        return report_text

        
if __name__ == "__main__":
   pass 