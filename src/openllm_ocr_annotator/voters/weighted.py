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
from src.openllm_ocr_annotator.voters.base import BaseVoter
from typing import List, Dict, Optional
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class WeightedVoter(BaseVoter):
    """Voter that uses configurable weights for different annotators and models."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize WeightingVoter with weights configuration.
        
        Args:
            weights: Dictionary mapping "annotator/model" to weight values.
                   Example: {
                       "OpenAIAnnotator/gpt-4-vision-preview": 1.0,
                       "OpenAIAnnotator/gpt-3.5-turbo-vision": 0.8,
                       "ClaudeAnnotator/claude-3-opus": 0.9
                   }
                   If not provided, all annotators will have equal weight of 1.0
        """
        self.weights = weights or {}
        self.default_weight = 1.0

    def get_weight(self, annotator_id: str) -> float:
        """Get weight for an annotator."""
        return self.weights.get(annotator_id, self.default_weight)

    def vote(self, annotations: List[Dict], annotator_ids: Optional[List[str]] = None) -> Dict:
        """Implement voting strategy by custom weighting.
        
        Args:
            annotations: List of annotation results from different annotators
            annotator_ids: List of annotator identifiers ("annotator/model") 
                         corresponding to each annotation
        
        Returns:
            Dict containing weighted voting results
        """
        if not annotations:
            raise ValueError("No annotations provided")
        
        # Use default annotator IDs if none provided
        if not annotator_ids:
            logger.warning("No annotator IDs provided, using model names as IDs")
            annotator_ids = [f"OpenAIAnnotator/{ann['model']}" for ann in annotations]
        
        if len(annotations) != len(annotator_ids):
            raise ValueError("Number of annotations and annotator IDs must match")

        # Initialize results structure
        voted_fields = defaultdict(lambda: defaultdict(float))
        
        # Process each annotation
        for annotation, annotator_id in zip(annotations, annotator_ids):
            # Get the weight for this annotator/model
            weight = self.get_weight(annotator_id)
            
            # Get fields from the result
            fields = annotation.get("result", {}).get("fields", [])
            
            # Process each field in the annotation
            for field in fields:
                field_name = field.get("field_name", None)
                field_value = field.get("value", None)
                field_confidence = field.get("confidence", 1.0)
                
                if not all([field_name, field_value]):
                    continue
                
                # Add weighted vote for this value
                # Weight is multiplied by confidence if available
                voted_fields[field_name][field_value] += weight * field_confidence
        
        # Compile final results
        results = {
            "fields": []
        }
        
        # For each field, select the value with highest weighted votes
        for field_name, value_votes in voted_fields.items():
            if not value_votes:
                continue
                
            # Get value with highest weighted votes
            best_value, best_weight = max(value_votes.items(), key=lambda x: x[1])
            
            # Calculate confidence as normalized weight
            total_weight = sum(value_votes.values())
            confidence = best_weight / total_weight if total_weight > 0 else 0
            
            results["fields"].append({
                "field_name": field_name,
                "value": best_value,
                "confidence": confidence
            })
        
        return results
