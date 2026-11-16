"""Experiment metadata for classical ML training runs.

Serializes training configuration, environment, metrics, and artifact paths
to JSON for auditability and reproducibility across training and inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class ExperimentMetadata:
    """Complete metadata for a classical ML training experiment."""

    experiment_id: str
    model_name: str
    algorithm: str
    task: str  # "binary" or "multiclass"
    dataset_name: Optional[str]
    dataset_version: Optional[str]
    feature_schema_version: Optional[str]
    random_seed: int
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    framework_versions: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: List[str] = field(default_factory=list)
    training_environment: str = "local"
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    finished_at: Optional[str] = None
    status: str = "running"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        return cls(**data)
