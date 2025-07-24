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
import logging
from typing import List, Dict
from src.openllm_ocr_annotator.voters.base import BaseVoter

logger = logging.getLogger(__name__)


class VotingManager:
    def __init__(self, annotator_infos: List[Dict[str, str]], voter: BaseVoter):
        """Initialize VotingManager.

        Args:
            annotator_infos: List of dicts containing annotator information, each with:
                - 'name': Annotator name (e.g. 'OpenAIAnnotator')
                - 'model': Model version (e.g. 'gpt-4-vision-preview')
            voter: Voting strategy to use
        """
        self.annotator_infos = annotator_infos
        self.voter = voter

    def collect_annotations(
        self, results_dir: Path, image_stem: str,num_samples:int=1
    ) -> Dict[str, Dict]:
        """Collect existing annotation results for an image.

        Args:
            results_dir: Directory containing annotation results
            image_stem: Image name without extension

        Returns:
            Dict mapping "annotator_name/model_version" to their results
        """
        results = {}
        for annotator_info in self.annotator_infos:
            annotator_name = annotator_info["name"]
            model_version = annotator_info.get("model", "default")
            if num_samples > 1:
                annotator_dir = [results_dir/annotator_name/model_version/"sampling"/f"sample_{str(i)}" for i in range(num_samples)]
                result_paths = [_d/ f"{image_stem}.json" for _d in annotator_dir]

            else:
                annotator_dir = results_dir / annotator_name / model_version
                result_path = annotator_dir / f"{image_stem}.json"

            try:
                if num_samples > 1:
                    for i,result_path in enumerate(result_paths):
                        result_key = f"{annotator_name}/{model_version}/sample_{str(i)}"
                        if result_path.exists():
                            with open(result_path, "r") as f:
                                results[result_key] = json.load(f)
                        else:
                            logger.warning(f"No annotation result found at: {result_path}")
                else:
                    if result_path.exists():
                        with open(result_path, "r") as f:
                            result_key = f"{annotator_name}/{model_version}"
                            results[result_key] = json.load(f)
                    else:
                        logger.warning(f"No annotation result found at: {result_path}")
            except Exception as e:
                logger.error(f"Error loading {result_path}: {e}")
                continue

        if not results:
            logger.warning(f"No annotation results found for {image_stem}")
        else:
            logger.debug(f"Found {len(results)} annotation results for {image_stem}")

        return results

    def get_voted_result(self, image_path: Path, output_dir: Path, num_samples: int = 1) -> Dict:
        """Get or compute voted result for a single image based on individual annotations.

        Args:
            image_path: Path to image file
            output_dir: Directory containing individual results

        Returns:
            Dict with voted result and metadata
        """
        voted_dir = output_dir / "voted_results"
        voted_dir.mkdir(parents=True, exist_ok=True)

        # Check if voted result already exists
        # if voted_path.exists():
        #    logger.info("Loading existing voted result")
        #    with open(voted_path, 'r') as f:
        #        return json.load(f)

        # Collect existing annotations
        results = self.collect_annotations(results_dir=output_dir, image_stem=image_path.stem,num_samples=num_samples)
        if not results:
            raise ValueError(f"No valid annotations found for {image_path.name}")

        # Run voting with annotator IDs
        annotator_ids = list(results.keys())
        annotations = list(results.values())

        voted_result = {
            "result": self.voter.vote(annotations, annotator_ids,num_samples),
            "metadata": {
                "annotators": annotator_ids,  # List of annotator_name/model_version used
            },
        }

        return voted_result
