"""Declarative schema definition and validation for ingested dataframes.

The schema is intentionally generic: it expresses column requirements and
optional type hints without assuming any particular dataset domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Schema:
    """Describes the expected shape of an ingested dataframe.

    Attributes:
        required_columns: Column names that must be present.
        optional_columns: Column names that may be present.
        column_types: Mapping of column name to a pandas-compatible dtype
            (e.g. ``"float64"``, ``"int64"``, ``"str"``). Columns listed here
            are cast during validation when possible.
        strict: When True, columns not in required or optional are reported
            as schema errors. When False (default), extra columns are allowed.
    """

    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    strict: bool = False

    def all_columns(self) -> List[str]:
        """Return the full set of known column names."""
        return list(self.required_columns) + list(self.optional_columns)

    def validate(self, df: pd.DataFrame) -> List[str]:
        """Validate a dataframe against the schema.

        Returns a list of human-readable error messages. An empty list means
        the dataframe is valid.
        """
        errors: List[str] = []
        columns = set(df.columns)

        missing = [c for c in self.required_columns if c not in columns]
        if missing:
            errors.append(
                "missing required columns: " + ", ".join(sorted(missing))
            )

        if self.strict:
            known = set(self.all_columns())
            unexpected = sorted(columns - known)
            if unexpected:
                errors.append(
                    "unexpected columns (strict mode): "
                    + ", ".join(unexpected)
                )

        # Attempt type casting for declared columns; report failures.
        for col, dtype in self.column_types.items():
            if col not in columns:
                continue
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"column '{col}' cannot be cast to {dtype}: {exc}"
                )

        return errors
