"""Ingestion result: a standardized dataframe plus per-file metadata.

The dataframe is standardized (whitespace-stripped column names, reset index)
but otherwise left domain-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .stats import FileStats


@dataclass
class IngestionResult:
    """Outcome of ingesting a single file."""

    file_path: str
    file_name: str
    dataframe: pd.DataFrame
    stats: FileStats
    encoding_used: str
    schema_errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.stats.valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "encoding_used": self.encoding_used,
            "rows": int(len(self.dataframe)),
            "columns": int(self.dataframe.shape[1]),
            "schema_errors": self.schema_errors,
            "valid": self.valid,
            "stats": self.stats.to_dict(),
        }
