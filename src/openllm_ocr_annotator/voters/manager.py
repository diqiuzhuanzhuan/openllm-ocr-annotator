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

    def annotate_with_all(self, image_path: str, output_dir: Path) -> Dict[str, Dict]:
        """Run annotation with all annotators and save individual results.
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save individual results
            
        Returns:
            Dict mapping annotator names to their results
        """
        results = {}
        img_path = Path(image_path)
        
        for annotator in self.annotators:
            annotator_name = annotator.__class__.__name__
            try:
                # Create directory for this annotator if not exists
                annotator_dir = output_dir / annotator_name
                annotator_dir.mkdir(parents=True, exist_ok=True)
                
                # Define output path for this image
                result_path = annotator_dir / f"{img_path.stem}.json"
                
                # If result exists, load it; otherwise annotate
                if result_path.exists():
                    logger.info(f"Loading existing result for {annotator_name}")
                    with open(result_path, 'r') as f:
                        results[annotator_name] = json.load(f)
                else:
                    logger.info(f"Running annotation with {annotator_name}")
                    result = annotator.annotate(str(img_path))
                    
                    # Save individual result
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    results[annotator_name] = result
                    
            except Exception as e:
                logger.error(f"Error with {annotator_name}: {e}")
                continue
                
        return results

    def get_voted_result(self, image_path: str, output_dir: Path) -> Dict:
        """Get or compute voted result from individual annotations.
        
        Args:
            image_path: Path to image file
            output_dir: Directory containing individual results
            
        Returns:
            Voted result
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
            
        # Run voting
        voted_result = self.voter.vote(list(results.values()))
        
        # Save voted result
        with open(voted_path, 'w') as f:
            json.dump(voted_result, f, indent=2, ensure_ascii=False)
            
        return voted_result