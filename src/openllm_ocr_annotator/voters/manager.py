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
import json
from pathlib import Path
from typing import List, Dict
import logging
from typing import List, Dict
from src.openllm_ocr_annotator.voters.base import BaseVoter
from src.openllm_ocr_annotator.annotators.base import BaseAnnotator

logger = logging.getLogger(__name__)

class VotingManager:
    def __init__(self, annotators: List[BaseAnnotator], voter: BaseVoter):
        self.annotators = annotators
        self.voter = voter
    
    def collect_annotations(self, results_dir: Path, image_stem: str) -> Dict[str, Dict]:
        """Collect existing annotation results for an image.
        
        Args:
            results_dir: Directory containing annotation results
            image_stem: Image name without extension
            
        Returns:
            Dict mapping "annotator_name/model_version" to their results
        """
        results = {}
        for annotator in self.annotators:
            annotator_name = annotator.__class__.__name__
            # Get model version from annotator if available, default to "default"
            model_version = getattr(annotator, "model", "default") 
            annotator_dir = results_dir / annotator_name / model_version
            result_path = annotator_dir / f"{image_stem}.json"
            
            try:
                if result_path.exists():
                    with open(result_path, 'r') as f:
                        result_key = f"{annotator_name}/{model_version}"
                        results[result_key] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {result_path}: {e}")
                continue
                
        if not results:
            logger.warning(f"No annotation results found for {image_stem}")
        else:
            logger.info(f"Found {len(results)} annotation results for {image_stem}")
            
        return results

    def annotate_with_all(self, image_path: str, output_dir: Path) -> Dict[str, Dict]:
        """Run annotation with all annotators and save individual results.
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save individual results
            
        Returns:
            Dict mapping "annotator_name/model_version" to their results
        """
        results = {}
        img_path = Path(image_path)
        
        for annotator in self.annotators:
            annotator_name = annotator.__class__.__name__
            # Get model version from annotator
            model_version = getattr(annotator, "model", "default")
            try:
                # Create directory for this annotator/model if not exists
                annotator_dir = output_dir / annotator_name / model_version
                annotator_dir.mkdir(parents=True, exist_ok=True)
                
                # Define output path for this image
                result_path = annotator_dir / f"{img_path.stem}.json"
                result_key = f"{annotator_name}/{model_version}"
                
                # If result exists, load it; otherwise annotate
                if result_path.exists():
                    logger.info(f"Loading existing result for {result_key}")
                    with open(result_path, 'r') as f:
                        results[result_key] = json.load(f)
                else:
                    logger.info(f"Running annotation with {result_key}")
                    result = annotator.annotate(str(img_path))
                    
                    # Save individual result
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    results[result_key] = result
                    
            except Exception as e:
                logger.error(f"Error with {annotator_name}/{model_version}: {e}")
                continue
                
        return results

    def get_voted_result(self, image_path: str, output_dir: Path) -> Dict:
        """Get or compute voted result from individual annotations.
        
        Args:
            image_path: Path to image file
            output_dir: Directory containing individual results
            
        Returns:
            Dict with voted result and metadata
        """
        img_path = Path(image_path)
        voted_dir = output_dir / "voted_results"
        voted_dir.mkdir(parents=True, exist_ok=True)
        voted_path = voted_dir / f"{img_path.stem}.json"
        
        # Check if voted result already exists
        if voted_path.exists():
            logger.info("Loading existing voted result")
            with open(voted_path, 'r') as f:
                return json.load(f)
        
        # Get all individual results
        results = self.annotate_with_all(image_path, output_dir)
        if not results:
            raise ValueError("No valid annotations to vote on")
            
        # Run voting with annotator IDs
        annotator_ids = list(results.keys())
        annotations = list(results.values())
        
        voted_result = {
            "result": self.voter.vote(annotations, annotator_ids),
            "metadata": {
                "annotators": annotator_ids  # List of annotator_name/model_version used
            }
        }
        
        # Save voted result
        with open(voted_path, 'w') as f:
            json.dump(voted_result, f, indent=2, ensure_ascii=False)
            
        return voted_result