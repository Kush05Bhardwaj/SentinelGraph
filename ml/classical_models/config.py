"""Configuration for classical ML model training.

All settings are explicit and serializable so training runs are deterministic
and reproducible across runs and environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a single classical ML model.

    Attributes:
        name: Short model identifier (e.g. ``"logistic_regression"``).
        algorithm: sklearn-compatible algorithm class name
            (e.g. ``"LogisticRegression"``, ``"RandomForestClassifier"``).
        hyperparameters: Keyword arguments passed to the model constructor.
        task: Classification task type - ``"binary"`` or ``"multiclass"``.
        random_seed: Seed for reproducibility.
    """

    name: str
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    task: str = "binary"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.task not in ("binary", "multiclass"):
            raise ModelConfigurationError(
                f"task must be 'binary' or 'multiclass', got {self.task!r}"
            )


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes:
        use_validation: Whether to evaluate on a held-out validation split.
        metric_primary: Primary metric for model selection/reporting
            (e.g. ``"f1_macro"``, ``"accuracy"``, ``"roc_auc"``).
        metrics: List of metrics to compute during evaluation.
    """

    use_validation: bool = True
    metric_primary: str = "f1_macro"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro", "f1_weighted", "roc_auc"])


@dataclass
class ClassicalTrainingConfig:
    """Top-level classical ML training configuration.

    Attributes:
        models: List of model configurations to train.
        training: Training loop configuration.
        output_dir: Directory for model artifacts and metadata.
        random_seed: Global random seed for reproducibility.
    """

    models: List[ModelConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "artifacts/classical_models"
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "name": m.name,
                    "algorithm": m.algorithm,
                    "hyperparameters": m.hyperparameters,
                    "task": m.task,
                    "random_seed": m.random_seed,
                }
                for m in self.models
            ],
            "training": {
                "use_validation": self.training.use_validation,
                "metric_primary": self.training.metric_primary,
                "metrics": self.training.metrics,
            },
            "output_dir": self.output_dir,
            "random_seed": self.random_seed,
        }
