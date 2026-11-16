"""File-level statistics captured during ingestion.

Statistics are descriptive only and never modify the source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FileStats:
    """Metrics for a single ingested file."""

    file_path: str
    file_name: str
    file_size_bytes: int
    encoding_used: str
    rows_read: int
    rows_valid: int
    rows_malformed: int
    column_count: int
    column_names: List[str]
    schema_errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when the file was ingested without schema or malformed-row errors."""
        return not self.schema_errors and self.rows_malformed == 0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "encoding_used": self.encoding_used,
            "rows_read": self.rows_read,
            "rows_valid": self.rows_valid,
            "rows_malformed": self.rows_malformed,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "schema_errors": self.schema_errors,
            "valid": self.valid,
        }
