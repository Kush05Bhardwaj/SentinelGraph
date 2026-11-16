"""Feature schema: documents every engineered feature and its derivation.

The schema is both human-readable and machine-consumable. It is produced
during ``fit()`` and carried through ``transform()`` so that training and
inference share an identical, documented feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureEntry:
    """Describes a single engineered feature."""

    name: str
    derivation: str  # human-readable derivation
    source_columns: List[str] = field(default_factory=list)
    dtype: str = "float64"
    is_derived: bool = True
    is_rate: bool = False
    is_frequency: bool = False
    notes: Optional[str] = None


@dataclass
class FeatureSchema:
    """Complete feature contract produced by the feature engineer."""

    version: str = "1.0"
    random_seed: int = 42
    input_columns: List[str] = field(default_factory=list)
    raw_columns: List[str] = field(
        default_factory=list
    )  # input columns carried through unchanged
    engineered_columns: List[str] = field(default_factory=list)
    feature_entries: List[FeatureEntry] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)
    excluded_leakage_columns: List[str] = field(default_factory=list)

    def feature_names(self) -> List[str]:
        """Ordered list of engineered feature names."""
        return [e.name for e in self.feature_entries]

    def all_output_columns(self) -> List[str]:
        """Raw columns + engineered columns (in stable order)."""
        return list(self.raw_columns) + self.feature_names()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "random_seed": self.random_seed,
            "input_columns": self.input_columns,
            "raw_columns": self.raw_columns,
            "engineered_columns": self.engineered_columns,
            "feature_entries": [
                {
                    "name": e.name,
                    "derivation": e.derivation,
                    "source_columns": e.source_columns,
                    "dtype": e.dtype,
                    "is_derived": e.is_derived,
                    "is_rate": e.is_rate,
                    "is_frequency": e.is_frequency,
                    "notes": e.notes,
                }
                for e in self.feature_entries
            ],
            "identifier_columns": self.identifier_columns,
            "excluded_leakage_columns": self.excluded_leakage_columns,
        }
