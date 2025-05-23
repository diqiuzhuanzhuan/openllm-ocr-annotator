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

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Set
import yaml
from pathlib import Path
from utils.logger import setup_logger
from enum import Enum

logger = setup_logger(__name__)

class EnsembleStrategy(Enum):
    # Options: weighted_vote, simple_vote, highest_confidence
    WEIGHTED_VOTE: str = "weighted_vote"# Options: weighted_vote, simple_vote, highest_confidence
    SIMPLE_VOTE: str = "simple_vote"  # Options: simple_vote, highest_confidence
    HIGHEST_CONFIDENCE: str = "highest_confidence"  # Options: highest_confidence, simple_vote, weighted_vote

    @classmethod
    def from_str(cls, value: str) -> "EnsembleStrategy":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown voting strategy: {value}. Must be one of {[s.value for s in cls]}")


@dataclass
class AnnotatorConfig:
    """Configuration for a single annotator"""
    name: str
    type: str
    task: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    weight: float = 1.0
    output_format: str = "json"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    enabled: bool = True
    prompt_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict) -> "AnnotatorConfig":
        """Load configuration from a dictionary"""
        return cls(**config)
            

@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting"""
    method: EnsembleStrategy
    min_confidence: float
    agreement_threshold: float
    output_format: str = "json"
    
    @classmethod
    def from_dict(cls, config: Dict) -> None:
        """Load configuration from a dictionary"""
        return cls(**config)

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    name: str = "default_dataset"
    version: str = 1.0
    description: str = ""
    format: str = "json"
    output_dir: str = "./datasets"
    split_ratio: float = 0.9
    num_samples: int = -1  # -1 means use all available samples

    @classmethod
    def from_dict(cls, config: Dict) -> "DatasetConfig":
        """Create DatasetConfig from a dictionary"""
        return cls(
            name=config.get("name", "default_dataset"),
            version=config.get("version", "1.0"),
            description=config.get("description", ""),
            format=config.get("format", "json"),
            output_dir=config.get("output_dir", "./datasets"),
            split_ratio=config.get("split_ratio", 0.8),
            num_samples=config.get("num_samples", -1)
        )
    
@dataclass
class TaskConfig:
    """Main task configuration"""
    task_id: str
    input_dir: str
    output_dir: str
    prompt_path: str
    annotators: List[AnnotatorConfig]
    ensemble: EnsembleConfig
    dataset: DatasetConfig
    max_files: int = -1  # -1 means no limit

    @classmethod
    def from_dict(cls, config: Dict) -> None:
        """Load configuration from a dictionary"""
        return cls(**config)


class AnnotatorConfigManager:
    """Manager for handling annotator configurations"""
    
    def __init__(self, config: Dict):
        self.version = config.get("version", "1.0")
        self.task = self._parse_task_config(config["task"])

    @classmethod
    def from_file(cls, config_path: str | Path) -> "AnnotatorConfigManager":
        """Create configuration manager from a YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            AnnotatorConfigManager instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        return cls(config)
    
    def _validate_config_keys(self, config_dict: Dict, dataclass_type: type, path: str = "") -> None:
        """Validate that all keys in the config dictionary are valid fields in the dataclass.
        
        Args:
            config_dict: Dictionary containing configuration
            dataclass_type: The dataclass type to validate against
            path: Current path in the config for nested structures
        """
        valid_fields = {field.name for field in fields(dataclass_type)}
        actual_fields = set(config_dict.keys())
        
        # Find unknown fields
        unknown_fields = actual_fields - valid_fields
        if unknown_fields:
            logger.warning(
                f"Unknown configuration fields found in {path or 'root'}: {unknown_fields}. "
                f"Valid fields are: {valid_fields}"
            )

    def _parse_task_config(self, task_config: Dict) -> TaskConfig:
        """Parse task configuration section.
        
        Args:
            task_config: Dictionary containing task configuration
            
        Returns:
            TaskConfig instance
        """
        # Validate main task config
        self._validate_config_keys(task_config, TaskConfig)

        # Parse and validate annotator configs
        annotators = []
        for i, ann_config in enumerate(task_config["annotators"]):
            # Validate annotator config
            self._validate_config_keys(
                ann_config, 
                AnnotatorConfig, 
                f"annotators[{i}]"
            )
            annotator = AnnotatorConfig.from_dict(ann_config)
            
            annotators.append(annotator)
            
        # Parse and validate ensemble config
        self._validate_config_keys(
            task_config["ensemble"], 
            EnsembleConfig, 
            "ensemble"
        )
        ensemble = EnsembleConfig(
            method=task_config["ensemble"]["method"],
            min_confidence=task_config["ensemble"]["min_confidence"],
            agreement_threshold=task_config["ensemble"]["agreement_threshold"],
            output_format=task_config["ensemble"].get("output_format", "json")
        )

        # Parse and validate dataset config
        dataset_config = task_config.get("dataset", {})
        self._validate_config_keys(
            dataset_config, 
            DatasetConfig, 
            "dataset"
        )
        dataset = DatasetConfig(
            name=dataset_config.get("name", task_config["task_id"]),
            version=dataset_config.get("version", "1.0"),
            description=dataset_config.get("description", ""),
            format=dataset_config.get("format", "json"),
            output_dir=dataset_config.get("output_dir", "./datasets"),
            split_ratio=dataset_config.get("split_ratio", 0.8),
            num_samples=dataset_config.get("num_samples", -1)
        )
        
        return TaskConfig(
            task_id=task_config.get("task_id", "default_task"),
            input_dir=task_config["input_dir"],
            output_dir=task_config["output_dir"],
            prompt_path=task_config.get("prompt_path"),
            max_files=task_config.get("max_files", -1),
            annotators=annotators,
            ensemble=ensemble,
            dataset=dataset
        )
    
    def get_enabled_annotators(self) -> List[AnnotatorConfig]:
        """Get list of enabled annotators.
        
        Returns:
            List of enabled AnnotatorConfig instances
        """
        return [ann for ann in self.task.annotators if ann.enabled]
    
    def get_annotator_weights(self) -> Dict[str, float]:
        """Get dictionary of annotator name to weight mapping.
        
        Returns:
            Dictionary mapping annotator names to their weights
        """
        return {
            f"{ann.name}/{ann.model}": ann.weight 
            for ann in self.task.annotators 
            if ann.enabled
        }

    def get_task_config(self) -> TaskConfig:
        """Get the main task configuration.
        
        Returns:
            TaskConfig instance
        """
        return self.task

    def get_ensemble_config(self) -> EnsembleConfig:
        """Get the ensemble configuration.
        
        Returns:
            EnsembleConfig instance
        """
        return self.task.ensemble

    def get_dataset_config(self) -> DatasetConfig:
        """Get the dataset configuration.
        
        Returns:
            DatasetConfig instance
        """
        return self.task.dataset

if __name__ == "__main__":
    # Example usage
    logger.setLevel("INFO")
    config_path = Path("examples")/"config.yaml"
    
    config_manager = AnnotatorConfigManager.from_file(config_path)
    enabled_annotators = config_manager.get_enabled_annotators()
    annotator_weights = config_manager.get_annotator_weights()
    
    logger.info("Enabled Annotators:")
    for ann in enabled_annotators:
        logger.info(f"- ann: {ann}")
    
    logger.info("\nAnnotator Weights:")
    for name, weight in annotator_weights.items():
        # 混合使用 f-string 和 format
        logger.info(f"Model {name} has weight: {weight}")