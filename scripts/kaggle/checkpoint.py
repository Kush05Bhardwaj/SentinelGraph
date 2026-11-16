"""Checkpoint save and restore utilities for Kaggle training.

Checkpoints are stored under /kaggle/working/ by default (Kaggle's writable
output directory) but a custom directory may be supplied. All functions are
safe to call when no GPU is present.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # PyTorch is optional in the test environment
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

DEFAULT_CHECKPOINT_DIR = "/kaggle/working/checkpoints"


@dataclass
class CheckpointMetadata:
    """Metadata describing a saved checkpoint."""

    checkpoint_id: str
    epoch: int
    step: int
    metric_value: Optional[float]
    metric_name: Optional[str]
    model_path: str
    optimizer_path: Optional[str]
    scheduler_path: Optional[str]
    created_at: str
    source_file: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def checkpoint_dir(run_id: Optional[str] = None, base_dir: str = DEFAULT_CHECKPOINT_DIR) -> str:
    """Return the checkpoint directory for a run, creating it if needed."""
    directory = os.path.join(base_dir, run_id) if run_id else base_dir
    _ensure_dir(directory)
    return directory


def save_checkpoint(
    run_id: str,
    epoch: int,
    step: int,
    model_state: Dict[str, Any],
    metric_value: Optional[float] = None,
    metric_name: Optional[str] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    base_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> CheckpointMetadata:
    """Save a model checkpoint and its metadata to disk.

    Args:
        run_id: Unique identifier for the training run.
        epoch: Current epoch index.
        step: Current global step index.
        model_state: Model state_dict (or any serializable object).
        metric_value: Optional validation metric value.
        metric_name: Optional name of the metric.
        optimizer_state: Optional optimizer state_dict.
        scheduler_state: Optional scheduler state_dict.
        extra: Optional additional metadata.
        base_dir: Root checkpoint directory.

    Returns:
        CheckpointMetadata describing the saved checkpoint.
    """
    import uuid

    directory = checkpoint_dir(run_id, base_dir)
    checkpoint_id = f"{run_id}-e{epoch}-s{step}"
    model_path = os.path.join(directory, f"{checkpoint_id}.pt")
    optimizer_path: Optional[str] = None
    scheduler_path: Optional[str] = None

    if _HAS_TORCH:
        torch.save(model_state, model_path)  # type: ignore[attr-defined]
        if optimizer_state is not None:
            optimizer_path = os.path.join(directory, f"{checkpoint_id}.optimizer.pt")
            torch.save(optimizer_state, optimizer_path)  # type: ignore[attr-defined]
        if scheduler_state is not None:
            scheduler_path = os.path.join(directory, f"{checkpoint_id}.scheduler.pt")
            torch.save(scheduler_state, scheduler_path)  # type: ignore[attr-defined]
    else:
        # Fallback: persist as JSON when torch is unavailable (for testing).
        with open(model_path, "w") as f:
            json.dump(model_state, f)

    metadata = CheckpointMetadata(
        checkpoint_id=checkpoint_id,
        epoch=epoch,
        step=step,
        metric_value=metric_value,
        metric_name=metric_name,
        model_path=model_path,
        optimizer_path=optimizer_path,
        scheduler_path=scheduler_path,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extra=extra or {},
    )

    meta_path = os.path.join(directory, f"{checkpoint_id}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "checkpoint_id": metadata.checkpoint_id,
                "epoch": metadata.epoch,
                "step": metadata.step,
                "metric_value": metadata.metric_value,
                "metric_name": metadata.metric_name,
                "model_path": metadata.model_path,
                "optimizer_path": metadata.optimizer_path,
                "scheduler_path": metadata.scheduler_path,
                "created_at": metadata.created_at,
                "extra": metadata.extra,
            },
            f,
            indent=2,
        )

    return metadata


def list_checkpoints(run_id: str, base_dir: str = DEFAULT_CHECKPOINT_DIR) -> List[CheckpointMetadata]:
    """List all checkpoints for a run sorted by epoch then step."""
    import glob

    directory = os.path.join(base_dir, run_id)
    if not os.path.isdir(directory):
        return []

    metadatas: List[CheckpointMetadata] = []
    for meta_path in sorted(glob.glob(os.path.join(directory, "*.meta.json"))):
        with open(meta_path) as f:
            data = json.load(f)
        metadatas.append(
            CheckpointMetadata(
                checkpoint_id=data["checkpoint_id"],
                epoch=data["epoch"],
                step=data["step"],
                metric_value=data.get("metric_value"),
                metric_name=data.get("metric_name"),
                model_path=data["model_path"],
                optimizer_path=data.get("optimizer_path"),
                scheduler_path=data.get("scheduler_path"),
                created_at=data["created_at"],
                extra=data.get("extra", {}),
            )
        )
    metadatas.sort(key=lambda m: (m.epoch, m.step))
    return metadatas


def load_checkpoint(
    checkpoint_id: str,
    run_id: str,
    base_dir: str = DEFAULT_CHECKPOINT_DIR,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and its metadata by checkpoint ID.

    Returns a dict with keys: metadata, model, optimizer, scheduler.
    Missing optimizer/scheduler are returned as None.
    """
    directory = os.path.join(base_dir, run_id)
    meta_path = os.path.join(directory, f"{checkpoint_id}.meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

    with open(meta_path) as f:
        metadata = json.load(f)

    model_path = metadata["model_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint model file not found: {model_path}")

    if _HAS_TORCH:
        model_state = torch.load(model_path, map_location=map_location)  # type: ignore[attr-defined]
    else:
        with open(model_path) as f:
            model_state = json.load(f)

    optimizer_state = None
    if metadata.get("optimizer_path") and os.path.isfile(metadata["optimizer_path"]):
        if _HAS_TORCH:
            optimizer_state = torch.load(metadata["optimizer_path"], map_location=map_location)  # type: ignore[attr-defined]
        else:
            with open(metadata["optimizer_path"]) as f:
                optimizer_state = json.load(f)

    scheduler_state = None
    if metadata.get("scheduler_path") and os.path.isfile(metadata["scheduler_path"]):
        if _HAS_TORCH:
            scheduler_state = torch.load(metadata["scheduler_path"], map_location=map_location)  # type: ignore[attr-defined]
        else:
            with open(metadata["scheduler_path"]) as f:
                scheduler_state = json.load(f)

    return {
        "metadata": metadata,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
    }


def restore_best_checkpoint(
    run_id: str,
    base_dir: str = DEFAULT_CHECKPOINT_DIR,
    metric_name: Optional[str] = None,
    higher_is_better: bool = True,
) -> Optional[Dict[str, Any]]:
    """Restore the checkpoint with the best recorded metric for a run.

    Returns None when no checkpoints exist or none match the metric filter.
    """
    checkpoints = list_checkpoints(run_id, base_dir)
    if not checkpoints:
        return None

    candidates = checkpoints
    if metric_name:
        candidates = [c for c in checkpoints if c.metric_name == metric_name and c.metric_value is not None]
    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.metric_value or float("-inf")) if higher_is_better else min(
        candidates, key=lambda c: c.metric_value or float("inf")
    )
    return load_checkpoint(best.checkpoint_id, run_id, base_dir)


def cleanup_checkpoints(
    run_id: str,
    keep_last: int = 3,
    base_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> List[str]:
    """Remove all but the ``keep_last`` most recent checkpoints for a run.

    Returns the list of removed checkpoint IDs.
    """
    checkpoints = list_checkpoints(run_id, base_dir)
    if len(checkpoints) <= keep_last:
        return []

    to_remove = checkpoints[: len(checkpoints) - keep_last]
    removed: List[str] = []
    for ck in to_remove:
        for path in (ck.model_path, ck.optimizer_path, ck.scheduler_path):
            if path and os.path.isfile(path):
                os.remove(path)
        meta_path = os.path.join(os.path.dirname(ck.model_path), f"{ck.checkpoint_id}.meta.json")
        if os.path.isfile(meta_path):
            os.remove(meta_path)
        removed.append(ck.checkpoint_id)
    return removed
