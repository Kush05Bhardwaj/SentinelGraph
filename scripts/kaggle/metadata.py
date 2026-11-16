"""Experiment metadata export utilities for Kaggle training.

Exports experiment configuration, environment report, and run metadata to a
JSON file under /kaggle/working/ (or a custom output directory). All functions
are side-effect-minimal and independently testable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

DEFAULT_OUTPUT_DIR = "/kaggle/working"


@dataclass
class ExperimentMetadata:
    """Complete metadata for a training experiment.

    Schema fields match docs/experiment-standards.md Section 1.1.
    """

    experiment_id: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    random_seed: Optional[int] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_environment: Optional[str] = None
    framework_versions: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    status: str = "running"
    notes: Optional[str] = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_metadata(
    metadata: ExperimentMetadata,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
) -> str:
    """Write experiment metadata as JSON and return the file path.

    Args:
        metadata: ExperimentMetadata instance to serialize.
        output_dir: Directory to write into (defaults to /kaggle/working).
        filename: Optional output filename; defaults to
            <experiment_id>.metadata.json.

    Returns:
        Absolute path to the written JSON file.
    """
    _ensure_dir(output_dir)
    if filename is None:
        filename = f"{metadata.experiment_id}.metadata.json"
    path = os.path.join(output_dir, filename)

    data = asdict(metadata)
    # Normalize timestamp if not set
    if not data["timestamp"]:
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_metadata(path: str) -> ExperimentMetadata:
    """Load experiment metadata from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return ExperimentMetadata(
        experiment_id=data["experiment_id"],
        model_name=data.get("model_name"),
        model_version=data.get("model_version"),
        dataset_name=data.get("dataset_name"),
        dataset_version=data.get("dataset_version"),
        feature_schema_version=data.get("feature_schema_version"),
        random_seed=data.get("random_seed"),
        hyperparameters=data.get("hyperparameters", {}),
        training_environment=data.get("training_environment"),
        framework_versions=data.get("framework_versions", {}),
        metrics=data.get("metrics", {}),
        artifact_paths=data.get("artifact_paths", []),
        timestamp=data.get("timestamp", ""),
        status=data.get("status", "running"),
        notes=data.get("notes"),
    )


def finalize_experiment(
    metadata: ExperimentMetadata,
    status: str = "completed",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
) -> str:
    """Mark an experiment finished and re-export its metadata.

    Convenience wrapper: sets finished_at and status, then exports.
    """
    metadata.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata.status = status
    return export_metadata(metadata, output_dir=output_dir, filename=filename)


def merge_environment_into_metadata(
    metadata: ExperimentMetadata,
    environment_report: Dict[str, Any],
) -> ExperimentMetadata:
    """Merge an environment detection report into experiment metadata."""
    metadata.environment = environment_report
    return metadata
