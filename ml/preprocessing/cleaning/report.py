"""Preprocessing report: documents every transformation applied to the data.

The report is both human-readable (``to_dict``/JSON) and machine-consumable.
It records which columns were dropped, transformed, scaled, or encoded so that
downstream consumers and auditors can trace every preprocessing decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnReport:
    """Report for a single column."""

    name: str
    original_dtype: Optional[str] = None
    transformed_dtype: Optional[str] = None
    dropped: bool = False
    reason: Optional[str] = None
    missing_count: int = 0
    missing_pct: float = 0.0
    inf_count: int = 0
    inf_pct: float = 0.0
    transformed: bool = False
    transformation: Optional[str] = None  # e.g. "scaled_standard", "encoded_label"
    notes: Optional[str] = None


@dataclass
class PreprocessingReport:
    """Full report of a fit/transform cycle."""

    fitted: bool = False
    n_rows_train: int = 0
    n_rows_transform: int = 0
    n_columns_input: int = 0
    n_columns_output: int = 0
    label_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    column_reports: List[ColumnReport] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_column_report(self, report: ColumnReport) -> None:
        self.column_reports.append(report)

    def add_transformation(self, description: str) -> None:
        self.transformations.append(description)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted": self.fitted,
            "n_rows_train": self.n_rows_train,
            "n_rows_transform": self.n_rows_transform,
            "n_columns_input": self.n_columns_input,
            "n_columns_output": self.n_columns_output,
            "label_column": self.label_column,
            "identifier_columns": self.identifier_columns,
            "dropped_columns": self.dropped_columns,
            "column_reports": [
                {
                    "name": r.name,
                    "original_dtype": r.original_dtype,
                    "transformed_dtype": r.transformed_dtype,
                    "dropped": r.dropped,
                    "reason": r.reason,
                    "missing_count": r.missing_count,
                    "missing_pct": r.missing_pct,
                    "inf_count": r.inf_count,
                    "inf_pct": r.inf_pct,
                    "transformed": r.transformed,
                    "transformation": r.transformation,
                    "notes": r.notes,
                }
                for r in self.column_reports
            ],
            "transformations": self.transformations,
            "config_summary": self.config_summary,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
        }
