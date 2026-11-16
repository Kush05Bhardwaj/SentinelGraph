"""Artifact persistence for preprocessing transformers.

Fitted scalers, encoders, and column mappings are saved to disk so that the
same preprocessing can be applied at inference time without re-fitting (which
would leak information from the full dataset).

Artifacts are written as JSON for human inspection plus a compact binary
payload when the optional ``joblib`` dependency is available. JSON-only mode
is always supported.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .report import PreprocessingReport


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Save a JSON-serializable dict to ``path``, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON dict from ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifacts(
    directory: str | Path,
    report: PreprocessingReport,
    fitted_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Persist preprocessing artifacts to ``directory``.

    Writes:
        - ``preprocessing_report.json`` (full transformation report)
        - ``fitted_state.json`` (fitted scaler/encoder parameters, JSON-safe)
        - ``fitted_state.pkl`` (binary payload via joblib, if available)

    Returns the list of written file paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: List[str] = []

    report_path = directory / "preprocessing_report.json"
    save_json(report_path, report.to_dict())
    written.append(str(report_path))

    state: Dict[str, Any] = {"fitted_state": fitted_state or {}}
    state_path = directory / "fitted_state.json"
    save_json(state_path, state)
    written.append(str(state_path))

    # Optional binary payload for complex fitted objects (e.g. sklearn scalers).
    try:  # pragma: no cover - depends on joblib availability
        import joblib  # type: ignore

        pkl_path = directory / "fitted_state.pkl"
        joblib.dump(fitted_state or {}, pkl_path)
        written.append(str(pkl_path))
    except Exception:
        pass

    return written


def load_artifacts(directory: str | Path) -> Dict[str, Any]:
    """Load preprocessing artifacts from ``directory``.

    Returns a dict with keys ``report`` and ``fitted_state``.
    """
    directory = Path(directory)
    report_path = directory / "preprocessing_report.json"
    state_path = directory / "fitted_state.json"

    if not report_path.is_file():
        raise FileNotFoundError(f"Preprocessing report not found: {report_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Fitted state not found: {state_path}")

    report = load_json(report_path)
    state = load_json(state_path)

    # Prefer the binary payload if present (contains richer objects).
    pkl_path = directory / "fitted_state.pkl"
    fitted_state: Optional[Dict[str, Any]] = None
    if pkl_path.is_file():
        try:  # pragma: no cover - depends on joblib availability
            import joblib  # type: ignore

            fitted_state = joblib.load(pkl_path)
        except Exception:
            fitted_state = None
    if fitted_state is None:
        fitted_state = state.get("fitted_state", {})

    return {"report": report, "fitted_state": fitted_state}
