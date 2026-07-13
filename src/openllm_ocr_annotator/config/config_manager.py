# SPDX-FileCopyrightText: 2025 Loong Ma
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path
from openllm_ocr_annotator.utils.logger import setup_logger
from enum import Enum

logger = setup_logger(__name__)


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


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

    name: str = "default_annotator"
    type: str = "curator"
    task: str = "ocr"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    weight: float = 1.0
    output_format: str = "json"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
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
            api_key=_first_not_none(config.get("api_key"), provider.get("api_key")),
            model=_first_not_none(config.get("model"), provider.get("model_name")),
            base_url=_first_not_none(
                config.get("base_url"), backend_params.get("base_url")
            ),
            weight=config.get("weight", 1.0),
            output_format=config.get("output_format", "json"),
            max_tokens=_first_not_none(
                config.get("max_tokens"), generation_params.get("max_tokens")
            ),
            temperature=_first_not_none(
                config.get("temperature"), generation_params.get("temperature")
            ),
            enabled=config.get("enabled", True),
            prompt_path=config.get("prompt_path", None),
            num_samples=config.get("num_samples", 1),
            backend=_first_not_none(config.get("backend"), provider.get("backend")),
            rpm=_first_not_none(
                config.get("rpm"), backend_params.get("max_requests_per_minute")
            ),
            tpm=_first_not_none(
                config.get("tpm"), backend_params.get("max_tokens_per_minute")
            ),
            estimated_input_tokens=config.get("estimated_input_tokens", None),
            request_timeout=_first_not_none(
                config.get("request_timeout"), backend_params.get("request_timeout")
            ),
            curator_working_dir=config.get("curator_working_dir", None),
            provider=provider or None,
            backend_params=backend_params or None,
            generation_params=generation_params or None,
        )


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting"""

    # Kept as Any in the Hydra schema so CLI values use the public lowercase names;
    # from_dict converts it to EnsembleStrategy for runtime use.
    method: Any = "weighted_vote"
    min_confidence: float = 0.0
    agreement_threshold: float = 0.0
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
    output_dir: str = "./datasets"
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
            output_dir=str(config.get("output_dir", "./datasets")),
            split_ratio=config.get("split_ratio", 0.8),
            num_samples=config.get("num_samples", -1),
            enabled=config.get("enabled", True),
        )


@dataclass
class TaskConfig:
    """Main task configuration"""

    task_id: str = "default_task"
    input_dir: str = ""
    output_dir: str = "./data/outputs"
    prompt_path: Optional[str] = None
    annotators: List[AnnotatorConfig] = field(default_factory=list)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    max_files: int = -1  # -1 means no limit
    num_samples: int = 1  # Number of samples per image

    @classmethod
    def from_dict(cls, config: Dict) -> "TaskConfig":
        """Load configuration from a dictionary"""
        return cls(**config)


@dataclass
class AppConfig:
    """Hydra's structured root configuration."""

    version: str = "1.0"
    task: TaskConfig = field(default_factory=TaskConfig)


def register_config_store() -> None:
    """Register the root schema before Hydra composes configuration groups."""
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="base_schema", node=AppConfig)


class AnnotatorConfigManager:
    """Manager for handling annotator configurations"""

    def __init__(self, config: Dict):
        self.version = config.get("version", "1.0")
        self.task = self._parse_task_config(self._normalize_task_config(config))

    @staticmethod
    def _to_structured_dict(config: Any) -> Dict:
        """Merge input with the structured schema and return resolved primitives."""
        from omegaconf import OmegaConf
        from omegaconf.errors import OmegaConfBaseException

        try:
            structured = OmegaConf.merge(OmegaConf.structured(AppConfig), config)
            resolved = OmegaConf.to_container(structured, resolve=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc
        if not isinstance(resolved, dict):
            raise TypeError("Configuration must resolve to a dictionary.")
        return resolved

    def _normalize_task_config(self, config: Dict) -> Dict:
        """Return the current task configuration shape."""
        if "task" in config:
            return config["task"]
        raise KeyError("Config must include 'task'.")

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

        return cls(cls._to_structured_dict(config))

    @classmethod
    def from_omegaconf(cls, config: Any) -> "AnnotatorConfigManager":
        """Create configuration manager from a Hydra/OmegaConf config."""
        return cls(cls._to_structured_dict(config))

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
            raise ValueError(
                f"Unknown configuration fields in {path or 'root'}: "
                f"{sorted(unknown_fields)}. Valid fields are: {sorted(valid_fields)}"
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
        ensemble = EnsembleConfig.from_dict(task_config["ensemble"])

        # Parse and validate dataset config
        dataset_config = task_config.get("dataset", {})
        self._validate_config_keys(dataset_config, DatasetConfig, "dataset")
        dataset_config.setdefault("name", task_config.get("task_id", "default_task"))
        dataset = DatasetConfig.from_dict(dataset_config)

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
