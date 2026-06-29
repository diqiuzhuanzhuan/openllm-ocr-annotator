# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path
from openllm_ocr_annotator.utils.logger import setup_logger
from enum import Enum

logger = setup_logger(__name__)


class EnsembleStrategy(Enum):
    # Options: weighted_vote, simple_vote, highest_confidence
    WEIGHTED_VOTE: str = (
        "weighted_vote"  # Options: weighted_vote, simple_vote, highest_confidence
    )
    SIMPLE_VOTE: str = "simple_vote"  # Options: simple_vote, highest_confidence
    HIGHEST_CONFIDENCE: str = (
        "highest_confidence"  # Options: highest_confidence, simple_vote, weighted_vote
    )

    @classmethod
    def from_str(cls, value: str) -> "EnsembleStrategy":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown voting strategy: {value}. Must be one of {[s.value for s in cls]}"
            )


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
    temperature: Optional[float] = 0
    enabled: bool = True
    prompt_path: Optional[str] = None
    num_samples: Optional[int] = (
        1  # To enable sampling, set num_samples > 1 and temperature between 0 and 1
    )
    backend: Optional[str] = None
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    estimated_input_tokens: Optional[int] = None
    request_timeout: Optional[int] = None
    curator_working_dir: Optional[str] = None
    provider: Optional[Dict] = None
    backend_params: Optional[Dict] = None
    generation_params: Optional[Dict] = None

    @classmethod
    def from_dict(cls, config: Dict) -> "AnnotatorConfig":
        """Load configuration from a dictionary"""
        provider = config.get("provider") or {}
        backend_params = (
            provider.get("backend_params") or config.get("backend_params") or {}
        )
        generation_params = (
            provider.get("generation_params") or config.get("generation_params") or {}
        )
        annotator_type = config.get("type", "curator")
        if annotator_type != "curator":
            raise ValueError(
                f"Unsupported annotator type: {annotator_type}. Only 'curator' is supported."
            )
        return cls(
            name=config.get("name", "default_annotator"),
            type=annotator_type,
            task=config.get("task", "ocr"),
            api_key=config.get("api_key", provider.get("api_key", None)),
            model=config.get("model", provider.get("model_name", None)),
            base_url=config.get("base_url", backend_params.get("base_url", None)),
            weight=config.get("weight", 1.0),
            output_format=config.get("output_format", "json"),
            max_tokens=config.get(
                "max_tokens", generation_params.get("max_tokens", None)
            ),
            temperature=config.get(
                "temperature", generation_params.get("temperature", None)
            ),
            enabled=config.get("enabled", True),
            prompt_path=config.get("prompt_path", None),
            num_samples=config.get("num_samples", 1),
            backend=config.get("backend", provider.get("backend", None)),
            rpm=config.get("rpm", backend_params.get("max_requests_per_minute", None)),
            tpm=config.get("tpm", backend_params.get("max_tokens_per_minute", None)),
            estimated_input_tokens=config.get("estimated_input_tokens", None),
            request_timeout=config.get(
                "request_timeout", backend_params.get("request_timeout", None)
            ),
            curator_working_dir=config.get("curator_working_dir", None),
            provider=provider or None,
            backend_params=backend_params or None,
            generation_params=generation_params or None,
        )


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting"""

    method: EnsembleStrategy
    min_confidence: float
    agreement_threshold: float
    output_format: str = "json"
    enabled: bool = True

    @classmethod
    def from_dict(cls, config: Dict) -> "EnsembleConfig":
        """Load configuration from a dictionary"""
        return cls(
            method=EnsembleStrategy.from_str(config.get("method", "weighted_vote")),
            min_confidence=config.get("min_confidence", 0.0),
            agreement_threshold=config.get("agreement_threshold", 0.0),
            output_format=config.get("output_format", "json"),
            enabled=config.get("enabled", True),
        )


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""

    name: str = "default_dataset"
    version: str = "1.0"
    description: str = ""
    format: str = "json"
    output_dir: str | Path = "./datasets"
    split_ratio: float = 0.9
    num_samples: int = -1  # -1 means use all available samples
    enabled: bool = True

    @classmethod
    def from_dict(cls, config: Dict) -> "DatasetConfig":
        """Create DatasetConfig from a dictionary"""
        return cls(
            name=config.get("name", "default_dataset"),
            version=config.get("version", "1.0"),
            description=config.get("description", ""),
            format=config.get("format", "json"),
            output_dir=Path(config.get("output_dir", "./datasets"))
            / Path(config.get("name", "default_dataset")),
            split_ratio=config.get("split_ratio", 0.8),
            num_samples=config.get("num_samples", -1),
            enabled=config.get("enabled", True),
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
    num_samples: int = 1  # Number of samples per image

    @classmethod
    def from_dict(cls, config: Dict) -> "TaskConfig":
        """Load configuration from a dictionary"""
        return cls(**config)


class AnnotatorConfigManager:
    """Manager for handling annotator configurations"""

    def __init__(self, config: Dict):
        self.version = config.get("version", "1.0")
        self.task = self._parse_task_config(self._normalize_task_config(config))

    def _normalize_task_config(self, config: Dict) -> Dict:
        """Normalize supported top-level config shapes into TaskConfig input."""
        if "task" in config:
            return config["task"]
        if "function_call_generation" in config:
            return self._function_call_generation_to_task(
                config["function_call_generation"]
            )
        raise KeyError(
            "Config must include either 'task' or 'function_call_generation'."
        )

    def _function_call_generation_to_task(self, config: Dict) -> Dict:
        provider = config.get("provider") or {}
        return {
            "task_id": config.get("name", "function_call_generation"),
            "input_dir": config.get("function_dataset"),
            "output_dir": config.get("output_dir", "data"),
            "prompt_path": config.get("prompt_path"),
            "max_files": config.get("max_num", -1),
            "annotators": [
                {
                    "name": config.get("name", "function_call_generation"),
                    "type": config.get("type", "curator"),
                    "task": config.get("task", "function_call_generation"),
                    "enabled": config.get("enable", True),
                    "output_format": config.get("output_format", "jsonl"),
                    "prompt_path": config.get("prompt_path"),
                    "provider": provider,
                }
            ],
            "ensemble": {
                "enabled": False,
                "method": "weighted_vote",
                "min_confidence": 0.0,
                "agreement_threshold": 0.0,
                "output_format": config.get("output_format", "jsonl"),
            },
            "dataset": {
                "enabled": False,
                "name": config.get("name", "function_call_generation"),
                "format": config.get("output_format", "jsonl"),
                "output_dir": config.get("output_dir", "data"),
                "num_samples": config.get("max_num", -1),
            },
        }

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

    @classmethod
    def from_omegaconf(cls, config: Any) -> "AnnotatorConfigManager":
        """Create configuration manager from a Hydra/OmegaConf config."""
        from omegaconf import OmegaConf

        resolved_config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved_config, dict):
            raise TypeError("OmegaConf config must resolve to a dictionary.")
        return cls(resolved_config)

    def _validate_config_keys(
        self, config_dict: Dict, dataclass_type: type, path: str = ""
    ) -> None:
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
        task_config = dict(task_config)

        # Validate main task config
        self._validate_config_keys(task_config, TaskConfig)

        # Parse and validate annotator configs
        annotators = []
        for i, ann_config in enumerate(task_config["annotators"]):
            # Validate annotator config
            self._validate_config_keys(ann_config, AnnotatorConfig, f"annotators[{i}]")
            annotator = AnnotatorConfig.from_dict(ann_config)
            if annotator.type != "curator":
                raise ValueError(
                    f"Unsupported annotator type: {annotator.type}. Only 'curator' is supported."
                )

            annotators.append(annotator)

        # Parse and validate ensemble config
        self._validate_config_keys(task_config["ensemble"], EnsembleConfig, "ensemble")
        ensemble = EnsembleConfig(
            method=EnsembleStrategy.from_str(task_config["ensemble"]["method"]),
            min_confidence=task_config["ensemble"]["min_confidence"],
            agreement_threshold=task_config["ensemble"]["agreement_threshold"],
            output_format=task_config["ensemble"].get("output_format", "json"),
            enabled=task_config["ensemble"].get("enabled", True),
        )

        # Parse and validate dataset config
        dataset_config = task_config.get("dataset", {})
        self._validate_config_keys(dataset_config, DatasetConfig, "dataset")
        dataset = DatasetConfig(
            name=dataset_config.get("name", task_config["task_id"]),
            version=dataset_config.get("version", "1.0"),
            description=dataset_config.get("description", ""),
            format=dataset_config.get("format", "json"),
            output_dir=dataset_config.get("output_dir", "./datasets"),
            split_ratio=dataset_config.get("split_ratio", 0.8),
            num_samples=dataset_config.get("num_samples", -1),
            enabled=dataset_config.get("enabled", True),
        )

        return TaskConfig(
            task_id=task_config.get("task_id", "default_task"),
            input_dir=task_config["input_dir"],
            output_dir=task_config["output_dir"],
            prompt_path=task_config.get("prompt_path"),
            max_files=task_config.get("max_files", -1),
            annotators=annotators,
            ensemble=ensemble,
            dataset=dataset,
            num_samples=task_config.get("num_samples", 1),
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
    config_path = Path("examples") / "config.yaml"

    config_manager = AnnotatorConfigManager.from_file(config_path)
    enabled_annotators = config_manager.get_enabled_annotators()
    annotator_weights = config_manager.get_annotator_weights()

    logger.info("Enabled Annotators:")
    for ann in enabled_annotators:
        logger.info(f"- ann: {ann}")

    logger.info("\nAnnotator Weights:")
    for name, weight in annotator_weights.items():
        # f-string formatting
        logger.info(f"Model {name} has weight: {weight}")
