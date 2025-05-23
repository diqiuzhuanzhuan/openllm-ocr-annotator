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

import argparse
from src import AnnotatorConfigManager
from src import run_batch_annotation

argparser = argparse.ArgumentParser(description="OpenLLM OCR Annotator")

argparser.add_argument(
    "--config",
    type=str,
    default="examples/config.yaml",
    help="Path to the annotator config file.",
)

if __name__ == "__main__":
    args = argparser.parse_args()
    from src.openllm_ocr_annotator.config.config_manager import AnnotatorConfigManager
    config_manager = AnnotatorConfigManager.from_file(args.config)
    
    annotator_configs = config_manager.get_enabled_annotators()
    weights = config_manager.get_annotator_weights()
    task_config = config_manager.get_task_config() 
    ensemble_config = config_manager.get_ensemble_config()
    dataset_config = config_manager.get_dataset_config()

    # Run batch annotation with weighted voting and create dataset
    # Here you can add the logic to load the configs and start the application
    # For example:
    # from openllm_ocr_annotator.config import AnnotatorConfig, DatasetConfig, EnsembleConfig
    # annotator_config = AnnotatorConfig(args.config)
    # dataset_config = DatasetConfig(args.dataset)
    # ensemble_config = EnsembleConfig(args.ensemble)
    # ... (rest of your application logic)
    task_config = config_manager.get_task_config() 
    run_batch_annotation(
        input_dir=task_config.input_dir,
        output_dir=task_config.output_dir,
        task_id=task_config.task_id,
        annotator_configs=annotator_configs,
        ensemble_strategy=ensemble_config.method,
        voting_weights=weights,
        max_files=task_config.max_files,
        create_dataset=True,
        dataset_split_ratio=dataset_config.split_ratio,
    )
    