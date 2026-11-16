"""Persistence for feature engineering configurations.

Saves and loads the feature schema and learned lookup tables (connection
frequency, source/destination statistics) to JSON so the same features can be
reproduced at inference time without re-fitting (which would leak
information from the full dataset).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import FeatureEntry, FeatureSchema


def save_config(
    directory: str | Path,
    schema: FeatureSchema,
    lookups: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Persist feature configuration to ``directory``.

    Writes:
        - ``feature_schema.json`` (full feature contract)
        - ``feature_lookups.json`` (learned lookups: connection frequencies,
          source/destination statistics)

    Returns the list of written file paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: List[str] = []

    schema_path = directory / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema.to_dict(), f, indent=2)
    written.append(str(schema_path))

    lookups = lookups or {}
    lookups_path = directory / "feature_lookups.json"
    with open(lookups_path, "w", encoding="utf-8") as f:
        json.dump(_lookups_to_json(lookups), f, indent=2)
    written.append(str(lookups_path))

    return written


def load_config(directory: str | Path) -> Dict[str, Any]:
    """Load feature configuration from ``directory``.

    Returns a dict with keys ``schema`` (FeatureSchema) and ``lookups``.
    """
    directory = Path(directory)
    schema_path = directory / "feature_schema.json"
    lookups_path = directory / "feature_lookups.json"

    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    if not lookups_path.is_file():
        raise FileNotFoundError(f"Feature lookups not found: {lookups_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)
    with open(lookups_path, "r", encoding="utf-8") as f:
        lookups_dict = json.load(f)

    schema = _schema_from_dict(schema_dict)
    lookups = _lookups_from_dict(lookups_dict)

    return {"schema": schema, "lookups": lookups}


def _lookups_to_json(lookups: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lookups (possibly containing tuples as keys) to JSON-safe form."""
    out: Dict[str, Any] = {}
    for key, value in lookups.items():
        if isinstance(key, tuple):
            key_str = "||".join(str(k) for k in key)
        else:
            key_str = str(key)
        out[key_str] = value
    return out


def _lookups_from_dict(lookups_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON-safe lookups back, attempting to restore tuple keys."""
    out: Dict[str, Any] = {}
    for key_str, value in lookups_dict.items():
        # Heuristic: keys containing "||" are tuple keys.
        if "||" in key_str:
            parts = key_str.split("||")
            out[tuple(parts)] = value
        else:
            out[key_str] = value
    return out


def _schema_from_dict(d: Dict[str, Any]) -> FeatureSchema:
    entries = [
        FeatureEntry(
            name=e["name"],
            derivation=e["derivation"],
            source_columns=list(e.get("source_columns", [])),
            dtype=e.get("dtype", "float64"),
            is_derived=bool(e.get("is_derived", True)),
            is_rate=bool(e.get("is_rate", False)),
            is_frequency=bool(e.get("is_frequency", False)),
            notes=e.get("notes"),
        )
        for e in d.get("feature_entries", [])
    ]
    return FeatureSchema(
        version=d.get("version", "1.0"),
        random_seed=int(d.get("random_seed", 42)),
        input_columns=list(d.get("input_columns", [])),
        raw_columns=list(d.get("raw_columns", [])),
        engineered_columns=list(d.get("engineered_columns", [])),
        feature_entries=entries,
        identifier_columns=list(d.get("identifier_columns", [])),
        excluded_leakage_columns=list(d.get("excluded_leakage_columns", [])),
    )
