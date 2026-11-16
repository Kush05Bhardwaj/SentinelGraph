"""Feature report: documents feature generation decisions per transform.

Records which input columns were present/missing, how many rows were
processed, and any warnings (e.g. zero-duration flows, division guards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureReport:
    """Outcome metadata for a fit or transform call."""

    fitted: bool = False
    n_rows: int = 0
    n_input_columns: int = 0
    n_engineered_features: int = 0
    input_columns: List[str] = field(default_factory=list)
    missing_input_columns: List[str] = field(default_factory=list)
    zero_duration_rows: int = 0
    division_guards_applied: int = 0
    transformations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_transformation(self, description: str) -> None:
        self.transformations.append(description)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted": self.fitted,
            "n_rows": self.n_rows,
            "n_input_columns": self.n_input_columns,
            "n_engineered_features": self.n_engineered_features,
            "input_columns": self.input_columns,
            "missing_input_columns": self.missing_input_columns,
            "zero_duration_rows": self.zero_duration_rows,
            "division_guards_applied": self.division_guards_applied,
            "transformations": self.transformations,
            "warnings": self.warnings,
        }
