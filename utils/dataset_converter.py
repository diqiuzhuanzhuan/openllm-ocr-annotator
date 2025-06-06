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


"""Convert voted annotation results to HuggingFace dataset format."""

import json
from pathlib import Path
from typing import Dict, List
from utils.logger import setup_logger
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image

logger = setup_logger(__name__)


def load_voted_results(voted_dir: str) -> List[Dict]:
    """Load all voted annotation results from directory.

    Args:
        voted_dir: Directory containing voted result JSON files

    Returns:
        List of annotation results
    """
    results = []
    voted_path = Path(voted_dir)

    for json_file in voted_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                result = json.load(f)
                # Add filename to metadata
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["filename"] = json_file.name
                results.append(result)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    return results


def create_hf_dataset(
    results: List[Dict],
    split_ratio: Dict[str, float] = {"train": 0.8, "test": 0.1, "validation": 0.1},
) -> DatasetDict:
    """Convert results to HuggingFace dataset format.

    Args:
        results: List of annotation results
        split_ratio: Dict defining train/test/validation split ratios

    Returns:
        DatasetDict containing the splits
    """
    # Validate split ratios
    if sum(split_ratio.values()) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    # Prepare features for the dataset
    features = Features(
        {
            "image": Image(),
            "image_path": Value("string"),
            "filename": Value("string"),
            "fields": Sequence(
                {
                    "field_name": Value("string"),
                    "value": Value("string"),
                    "confidence": Value("float32"),
                }
            ),
            "metadata": {
                "task_id": Value("string"),
                "timestamp": Value("string"),
                "annotators": Sequence(Value("string")),
            },
        }
    )

    # Convert results to dataset format
    dataset_dicts = []
    for result in results:
        image_path = result["metadata"]["image_path"]
        dataset_dict = {
            "image": image_path,
            "image_path": image_path,
            "filename": result["metadata"]["filename"],
            "fields": [
                {
                    "field_name": field["field_name"],
                    "value": field["value"],
                    "confidence": field["confidence"],
                }
                for field in result["result"]["fields"]
            ],
            "metadata": {
                "task_id": result["metadata"]["task_id"],
                "timestamp": result["metadata"]["timestamp"],
            },
        }
        dataset_dicts.append(dataset_dict)

    # Create the dataset
    full_dataset = Dataset.from_list(dataset_dicts, features=features)

    # Split the dataset
    splits = full_dataset.train_test_split(
        test_size=split_ratio["test"] + split_ratio.get("validation", 0),
        shuffle=True,
        seed=42,
    )

    # Further split test into test and validation
    if "validation" in split_ratio:
        val_size = split_ratio["validation"] / (
            split_ratio["test"] + split_ratio["validation"]
        )
        test_splits = splits["test"].train_test_split(
            test_size=val_size, shuffle=True, seed=42
        )
        return DatasetDict(
            {
                "train": splits["train"],
                "test": test_splits["train"],
                "validation": test_splits["test"],
            }
        )

    return DatasetDict(splits)


def convert_to_hf_dataset(
    voted_dir: str,
    output_dir: str,
    split_ratio: Dict[str, float] = {"train": 0.8, "test": 0.1, "validation": 0.1},
) -> DatasetDict:
    """Convert voted results to HuggingFace dataset and save it.

    Args:
        voted_dir: Directory containing voted results
        output_dir: Directory to save the dataset
        split_ratio: Dict defining train/test/validation split ratios

    Returns:
        DatasetDict containing the splits
    """
    # Load voted results
    results = load_voted_results(voted_dir)
    if not results:
        raise ValueError(f"No valid results found in {voted_dir}")

    # Create dataset
    dataset = create_hf_dataset(results, split_ratio)

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)

    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Train size: {len(dataset['train'])}")
    logger.info(f"Test size: {len(dataset['test'])}")
    if "validation" in dataset:
        logger.info(f"Validation size: {len(dataset['validation'])}")

    return dataset
