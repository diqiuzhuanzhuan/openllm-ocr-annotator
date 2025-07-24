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
import multiprocessing as mp
from typing import List
from src.openllm_ocr_annotator.pipeline.annotator_processor import AnnotatorProcessor
from src.openllm_ocr_annotator.config import AnnotatorConfig

logger = setup_logger(__name__)


class ParallelProcessor:
    """Manages parallel processing of multiple annotators."""

    def __init__(
        self,
        annotator_configs: List[AnnotatorConfig],
        output_dir: Path,
        max_workers: int = 8,
    ):
        """Initialize with annotator configurations instead of instances.

        Args:
            annotator_configs: List of dictionaries containing annotator configurations
            output_dir: Base output directory for results
        """
        self.annotator_configs = annotator_configs
        self.output_dir = output_dir
        self.max_workers = max_workers

    @staticmethod
    def create_annotator(config: AnnotatorConfig):
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

    def run_annotator_process(self, config: AnnotatorConfig, image_files: List[Path]):
        """Run single annotator process with fresh annotator instance."""
        try:
            # Create new annotator instance in this process
            processor = AnnotatorProcessor(
                annotator_config=config, output_dir=self.output_dir, max_workers=self.max_workers
            )
            processor.process_images(image_files=image_files)

        except Exception as e:
            logger.error(f"Error in annotator process: {e}")
            raise

    def run_parallel(self, image_files: List[Path]):
        """Run all annotators in parallel."""
        processes = []

        for config in self.annotator_configs:
            p = mp.Process(
                target=self.run_annotator_process, args=(config, image_files)
            )
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()
