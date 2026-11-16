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
    """Complete metadata for a training experiment."""

    experiment_id: str
    run_name: str
    source_dataset: Optional[str]
    model_architecture: Optional[str]
    task: Optional[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_ids: list = field(default_factory=list)
    artifact_paths: list = field(default_factory=list)
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finished_at: Optional[str] = None
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
    # Normalize finished_at/status if not set
    if data["finished_at"] is None:
        data["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_metadata(path: str) -> ExperimentMetadata:
    """Load experiment metadata from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return ExperimentMetadata(
        experiment_id=data["experiment_id"],
        run_name=data["run_name"],
        source_dataset=data.get("source_dataset"),
        model_architecture=data.get("model_architecture"),
        task=data.get("task"),
        hyperparameters=data.get("hyperparameters", {}),
        environment=data.get("environment", {}),
        metrics=data.get("metrics", {}),
        checkpoint_ids=data.get("checkpoint_ids", []),
        artifact_paths=data.get("artifact_paths", []),
        started_at=data.get("started_at", ""),
        finished_at=data.get("finished_at"),
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
    metadata.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata.status = status
    return export_metadata(metadata, output_dir=output_dir, filename=filename)


def merge_environment_into_metadata(
    metadata: ExperimentMetadata,
    environment_report: Dict[str, Any],
) -> ExperimentMetadata:
    """Merge an environment detection report into experiment metadata."""
    metadata.environment = environment_report
    return metadata
