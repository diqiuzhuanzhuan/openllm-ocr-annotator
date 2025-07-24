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


from utils.logger import setup_logger
from pathlib import Path
import json
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import time
from src.openllm_ocr_annotator.config import AnnotatorConfig
from src.openllm_ocr_annotator.annotators.base import BaseAnnotator
from utils.formatter import parse_json_from_text

logger = setup_logger(__name__)


def create_annotator(config: AnnotatorConfig)->"BaseAnnotator":
    """Create a new annotator instance from config."""
    if config.type == "openai":
        from src.openllm_ocr_annotator.annotators.openai_annotator import (
            OpenAIAnnotator,
        )

        return OpenAIAnnotator.from_config(config=config)
    elif config.type == "claude":
        from src.openllm_ocr_annotator.annotators.claude_annotator import (
            ClaudeAnnotator,
        )

        return ClaudeAnnotator(
            api_key=config["api_key"],
            model=config.get("model"),
            base_url=config.get("base_url", None),
        )
    elif config.type == "gemini":
        from src.openllm_ocr_annotator.annotators.gemini_annotator import (
            GeminiAnnotator,
        )

        return GeminiAnnotator.from_config(config=config)
    else:
        raise ValueError(f"Unknown annotator type: {config['type']}")


class AnnotatorProcessor:
    def __init__(
        self, annotator_config: AnnotatorConfig, output_dir: Path, max_workers: int = 8
    ):
        """Initialize the processor with an annotator and output directory.

        Args:
            annotator: The annotator to process images with
            output_dir: Base output directory for results
            max_workers: Maximum number of parallel workers (default: 8)
            num_samples: Number of samples to generate per image (default: 1)
            sampling_temperature: Temperature for sampling (default: None)
        """
        self.annotator = create_annotator(annotator_config)
        # Create output directory structure
        model_version = getattr(self.annotator, "model", "default")
        name = getattr(self.annotator, "name", "openai")

        # If using sampling, create subdirectories for each sample
        self.num_samples = annotator_config.num_samples
        # self.sampling_temperature = sampling_temperature
        self.sampling_temperature = annotator_config.temperature

        if self.num_samples > 1:
            # For sampling mode: /output_dir/model/version/sampling_{temperature}/sample_{i}/
            sampling_dir = (
                f"sampling_{self.sampling_temperature}"
                if self.sampling_temperature
                else "sampling"
            )
            self.output_dir = (
                output_dir
                / name.lstrip("/")
                / model_version.lstrip("/")
                / sampling_dir.lstrip("/")
            )
            self.sample_dirs = [
                self.output_dir / f"sample_{i}" for i in range(self.num_samples)
            ]
            for dir in self.sample_dirs:
                dir.mkdir(parents=True, exist_ok=True)
        else:
            # For single result mode: /output_dir/model/version/
            self.output_dir = output_dir / name.lstrip("/") / model_version.lstrip("/")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dirs = [self.output_dir]

        self.max_workers = max_workers

    def get_output_dir(self) -> Path:
        """Get the output directory for the current annotator."""
        return self.output_dir

    def _process_images(self, image_files: List[Path]):
        """Process a list of images with the annotator."""
        """
        Args:
            image_files (List[Path]): List of image file paths to process.
        """
        for img_path in tqdm(
            image_files, desc=f"Processing with {self.annotator.name}", unit="image"
        ):
            self.process_single_image(str(img_path))

    def process_images(self, image_files: List[Path]):
        """Asynchronously process a list of images with the annotator."""
        """
        Args:
            image_files (List[Path]): List of image file paths to process.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_single_image, img_path)
                for img_path in image_files
            ]
            _ = [
                future.result()
                for future in tqdm(
                    futures, desc=f"Processing with {self.annotator.name}", unit="image"
                )
            ]

    def _parse_and_validate_result(
        self, result: Union[str, Dict], img_path: Path, sample_id: Optional[int] = None
    ) -> Optional[Dict]:
        """Parse and validate annotation result.

        Args:
            result: Raw annotation result
            img_path: Path to the processed image
            sample_id: Optional sample ID for sampling mode

        Returns:
            Parsed and validated result dictionary or None if invalid
        """
        try:
            # Parse string result
            if isinstance(result, str):
                result = {"result": parse_json_from_text(result)}
            elif isinstance(result, dict) and "result" in result:
                # Parse nested result
                if isinstance(result["result"], list):
                    result["result"] = parse_json_from_text(result["result"][0])

            # Validate result
            if not result.get("result"):
                sample_info = f" (sample {sample_id})" if sample_id is not None else ""
                logger.warning(
                    f"No valid annotation found for {img_path.name}{sample_info}"
                )
                return None

            # Add metadata
            result["metadata"] = {
                "timestamp": int(time.time()),
                **(
                    {"sample_id": sample_id, "temperature": self.sampling_temperature}
                    if sample_id is not None
                    else {}
                ),
            }

            return result

        except Exception as e:
            sample_info = f" (sample {sample_id})" if sample_id is not None else ""
            logger.error(f"Error parsing result for {img_path}{sample_info}: {e}")
            return None

    def _save_result(self, result: Dict, save_path: Path) -> None:
        """Save result to JSON file.

        Args:
            result: Result dictionary to save
            save_path: Path where to save the result
        """
        try:
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving result to {save_path}: {e}")

    def process_single_image(
        self, image_path: Path
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Process single image and save result(s).

        Args:
            image_path: Path to the image file

        Returns:
            In single mode: Processed result dictionary or None if failed
            In sampling mode: List of result dictionaries or None if failed
        """
        if self.num_samples > 1:
            return self._process_sampling_mode(image_path)
        else:
            return self._process_single_mode(image_path)

    def _process_single_mode(self, img_path: Path) -> Optional[Dict]:
        """Process image in single result mode."""
        result_path = self.output_dir / f"{img_path.stem}.json"

        # Return cached result if exists
        if result_path.exists():
            try:
                with open(result_path, "r") as f:
                    result = json.load(f)
                    annotation = result.get("result",None)
                    if annotation:
                        return result
            except Exception as e:
                logger.error(f"Error loading cached result for {img_path}: {e}")

        try:
            # Get annotation
            result = self.annotator.annotate(str(img_path))

            # Parse and validate
            processed_result = self._parse_and_validate_result(result, img_path)
            if not processed_result:
                return None

            # Save result
            self._save_result(processed_result, result_path)
            return processed_result

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None

    def _process_sampling_mode(self, img_path: Path) -> Optional[List[Dict]]:
        """Process image in sampling mode."""
        results = []

        # Check for existing samples
        for i, sample_dir in enumerate(self.sample_dirs):
            result_path = sample_dir / f"{img_path.stem}.json"
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        results.append(json.load(f))
                except Exception as e:
                    logger.error(f"Error loading cached sample {i} for {img_path}: {e}")

        # Return if all samples exist
        if len(results) == self.num_samples:
            return results

        # Clear partial results to process all samples
        results = []

        try:
            # Get samples
            raw_results = self.annotator.annotate(str(img_path))
            meta_info = {k: raw_results[k] for k in raw_results if k != "result"}

            # Process each sample
            for i, raw_result in enumerate(raw_results.get("result", [])):
                processed_result = self._parse_and_validate_result(
                    raw_result, img_path, i
                )
                if processed_result:
                    # Save sample
                    processed_result.update(meta_info)
                    save_path = self.sample_dirs[i] / f"{img_path.stem}.json"
                    self._save_result(processed_result, save_path)
                    results.append(processed_result)

            return results if results else None

        except Exception as e:
            logger.error(f"Error processing samples for {img_path}: {e}")
            return None
