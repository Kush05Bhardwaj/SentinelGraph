"""CSV dataset ingestion for SentinelGraph.

Provides a generic, domain-agnostic CSV ingestor that:

* reads single or multiple CSV files from configurable local or Kaggle
  input directories,
* validates against an optional declarative schema,
* tries a primary encoding with configurable fallbacks,
* handles malformed rows (skip / warn / error),
* logs every ingested file,
* captures per-file statistics,
* returns a standardized dataframe plus metadata,
* never modifies raw source files.

The ingestor makes no dataset-specific assumptions: column names, types and
the label column are all driven by the supplied :class:`Schema` (or left
unspecified).
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .exceptions import (
    IngestionConfigurationError,
    IngestionEncodingError,
    IngestionEmptyFileError,
    IngestionError,
    IngestionFileNotFoundError,
    IngestionSchemaError,
)
from .logger import get_logger
from .result import IngestionResult
from .schema import Schema
from .stats import FileStats

DEFAULT_ENCODING = "utf-8"
DEFAULT_FALLBACK_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "latin-1", "cp1252")
DEFAULT_CHUNK_PROBE_BYTES = 8 * 1024 * 1024  # 8 MB encoding probe cap


@dataclass
class IngestorConfig:
    """Configuration shared by :class:`CSVIngestor` instances."""

    input_dir: Optional[str] = None
    schema: Optional[Schema] = None
    encoding: str = DEFAULT_ENCODING
    fallback_encodings: Tuple[str, ...] = DEFAULT_FALLBACK_ENCODINGS
    on_malformed: str = "skip"  # 'skip' | 'warn' | 'error'
    encoding_probe_bytes: int = DEFAULT_CHUNK_PROBE_BYTES
    logger: Optional[logging.Logger] = None


class CSVIngestor:
    """Generic CSV ingestor with schema validation and encoding handling."""

    def __init__(
        self,
        input_dir: Optional[str] = None,
        schema: Optional[Schema] = None,
        encoding: str = DEFAULT_ENCODING,
        fallback_encodings: Optional[Sequence[str]] = None,
        on_malformed: str = "skip",
        encoding_probe_bytes: int = DEFAULT_CHUNK_PROBE_BYTES,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if on_malformed not in ("skip", "warn", "error"):
            raise IngestionConfigurationError(
                f"on_malformed must be 'skip', 'warn' or 'error', got {on_malformed!r}"
            )

        self.config = IngestorConfig(
            input_dir=input_dir,
            schema=schema,
            encoding=encoding,
            fallback_encodings=tuple(
                fallback_encodings if fallback_encodings is not None
                else DEFAULT_FALLBACK_ENCODINGS
            ),
            on_malformed=on_malformed,
            encoding_probe_bytes=encoding_probe_bytes,
            logger=logger or get_logger(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, path: str) -> IngestionResult:
        """Ingest a single CSV file.

        Args:
            path: Filesystem path to the CSV file (local or Kaggle input).

        Returns:
            IngestionResult containing the standardized dataframe and stats.

        Raises:
            IngestionFileNotFoundError: if the file does not exist.
            IngestionEmptyFileError: if the file is empty.
            IngestionEncodingError: if no candidate encoding decodes it.
            IngestionSchemaError: if schema validation fails or (when
                on_malformed='error') malformed rows are present.
        """
        resolved = os.path.abspath(path)
        self.config.logger.debug("Ingesting file: %s", resolved)

        if not os.path.exists(resolved):
            raise IngestionFileNotFoundError(f"Input file not found: {resolved}")

        file_size = os.path.getsize(resolved)
        if file_size == 0:
            raise IngestionEmptyFileError(f"Input file is empty: {resolved}")

        encoding = self._resolve_encoding(resolved)
        dataframe, rows_read, rows_malformed = self._read_csv(resolved, encoding)

        # Standardize: strip whitespace from column names, reset index.
        dataframe = dataframe.copy()
        dataframe.columns = [str(c).strip() for c in dataframe.columns]
        dataframe = dataframe.reset_index(drop=True)

        # Schema validation (casts types where declared).
        schema_errors: List[str] = []
        if self.config.schema is not None:
            schema_errors = self.config.schema.validate(dataframe)
            if schema_errors:
                raise IngestionSchemaError(
                    f"Schema validation failed for {resolved}: "
                    + "; ".join(schema_errors)
                )

        rows_valid = len(dataframe)
        stats = FileStats(
            file_path=resolved,
            file_name=os.path.basename(resolved),
            file_size_bytes=file_size,
            encoding_used=encoding,
            rows_read=rows_read,
            rows_valid=rows_valid,
            rows_malformed=rows_malformed,
            column_count=int(dataframe.shape[1]),
            column_names=list(dataframe.columns),
            schema_errors=schema_errors,
        )

        self.config.logger.info(
            "Ingested %s | encoding=%s rows=%d valid=%d malformed=%d",
            os.path.basename(resolved), encoding, rows_read, rows_valid,
            rows_malformed,
        )

        return IngestionResult(
            file_path=resolved,
            file_name=os.path.basename(resolved),
            dataframe=dataframe,
            stats=stats,
            encoding_used=encoding,
            schema_errors=schema_errors,
        )

    def ingest_directory(
        self,
        input_dir: Optional[str] = None,
        pattern: str = "*.csv",
        recursive: bool = False,
        combine: bool = True,
    ) -> Tuple[List[IngestionResult], Optional[pd.DataFrame]]:
        """Ingest every CSV matching ``pattern`` under ``input_dir``.

        Args:
            input_dir: Directory to search. Defaults to the constructor's
                ``input_dir`` (supports Kaggle ``/kaggle/input/...`` paths).
            pattern: Glob pattern for CSV files.
            recursive: Whether to search subdirectories.
            combine: When True, return a concatenated dataframe of all
                successfully ingested files as the second element.

        Returns:
            (results, combined_dataframe). ``combined_dataframe`` is None
            when ``combine`` is False or no files were ingested.
        """
        directory = input_dir or self.config.input_dir
        if directory is None:
            raise IngestionConfigurationError(
                "No input_dir provided and none configured on the ingestor."
            )

        if not os.path.isdir(directory):
            raise IngestionFileNotFoundError(
                f"Input directory not found: {directory}"
            )

        glob_pattern = os.path.join(directory, "**", pattern) if recursive \
            else os.path.join(directory, pattern)
        files = sorted(glob.glob(glob_pattern, recursive=recursive))

        self.config.logger.info(
            "Found %d CSV files matching %r in %s",
            len(files), pattern, directory,
        )

        results: List[IngestionResult] = []
        frames: List[pd.DataFrame] = []

        for file_path in files:
            try:
                result = self.ingest_file(file_path)
            except IngestionError as exc:
                self.config.logger.warning(
                    "Skipping %s: %s", file_path, exc
                )
                continue
            results.append(result)
            frames.append(result.dataframe)

        combined: Optional[pd.DataFrame] = None
        if combine and frames:
            combined = pd.concat(frames, ignore_index=True)
            self.config.logger.info(
                "Combined %d frames into %d total rows",
                len(frames), len(combined),
            )

        return results, combined

    def ingest_all(
        self,
        input_dir: Optional[str] = None,
        pattern: str = "*.csv",
        recursive: bool = False,
    ) -> Tuple[List[IngestionResult], Optional[pd.DataFrame]]:
        """Alias for :meth:`ingest_directory`."""
        return self.ingest_directory(
            input_dir=input_dir, pattern=pattern,
            recursive=recursive, combine=True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_encoding(self, path: str) -> str:
        """Return the first candidate encoding that decodes the file."""
        candidates = [self.config.encoding] + list(
            e for e in self.config.fallback_encodings
            if e != self.config.encoding
        )

        probe = self._probe_encoding(path, candidates)
        if probe is not None:
            return probe

        # As a final safety net, attempt a full pandas read with each
        # encoding; the last-resort failure raises IngestionEncodingError.
        last_error: Optional[Exception] = None
        for enc in candidates:
            try:
                pd.read_csv(path, encoding=enc, nrows=1)
                return enc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        raise IngestionEncodingError(
            f"None of the candidate encodings {candidates} could decode {path}; "
            f"last error: {last_error}"
        )

    def _probe_encoding(
        self, path: str, candidates: Sequence[str]
    ) -> Optional[str]:
        """Probe candidate encodings by reading up to ``probe_bytes``.

        Returns the first encoding that decodes the probed bytes without a
        ``UnicodeDecodeError``, or None when none succeed.
        """
        cap = self.config.encoding_probe_bytes
        for enc in candidates:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    remaining = cap
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                return enc
            except UnicodeDecodeError:
                continue
        return None

    def _read_csv(
        self, path: str, encoding: str
    ) -> Tuple[pd.DataFrame, int, int]:
        """Read a CSV with malformed-row handling.

        Returns (dataframe, rows_read, rows_malformed).
        """
        # encoding is passed explicitly; do not also put it in kwargs.
        kwargs: Dict[str, Any] = {}

        # Attempt a strict parse first to detect malformed rows.
        try:
            strict = pd.read_csv(
                path, encoding=encoding, on_bad_lines="error", **kwargs
            )
            rows_read = len(strict)
            return strict, rows_read, 0
        except pd.errors.ParserError:
            pass  # malformed rows present; handle below

        total_lines = self._count_lines(path, encoding)
        rows_total = max(total_lines - 1, 0)  # subtract header

        if self.config.on_malformed == "error":
            raise IngestionSchemaError(
                f"Malformed rows detected in {path} and on_malformed='error'."
            )

        bad = "warn" if self.config.on_malformed == "warn" else "skip"
        dataframe = pd.read_csv(
            path, encoding=encoding, on_bad_lines=bad, **kwargs
        )
        rows_read = len(dataframe)
        rows_malformed = max(rows_total - rows_read, 0)

        if self.config.on_malformed == "warn" and rows_malformed > 0:
            self.config.logger.warning(
                "%d malformed row(s) skipped in %s", rows_malformed, path
            )

        return dataframe, rows_read, rows_malformed

    @staticmethod
    def _count_lines(path: str, encoding: str) -> int:
        """Count physical lines in a text file."""
        count = 0
        with open(path, "r", encoding=encoding, newline="") as f:
            for _ in f:
                count += 1
        return count


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def ingest_csv(
    path: str,
    *,
    schema: Optional[Schema] = None,
    encoding: str = DEFAULT_ENCODING,
    fallback_encodings: Optional[Sequence[str]] = None,
    on_malformed: str = "skip",
    logger: Optional[logging.Logger] = None,
) -> IngestionResult:
    """Ingest a single CSV file (convenience wrapper)."""
    ingestor = CSVIngestor(
        schema=schema, encoding=encoding,
        fallback_encodings=fallback_encodings,
        on_malformed=on_malformed, logger=logger,
    )
    return ingestor.ingest_file(path)


def ingest_directory(
    input_dir: str,
    *,
    pattern: str = "*.csv",
    schema: Optional[Schema] = None,
    encoding: str = DEFAULT_ENCODING,
    fallback_encodings: Optional[Sequence[str]] = None,
    on_malformed: str = "skip",
    recursive: bool = False,
    combine: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[IngestionResult], Optional[pd.DataFrame]]:
    """Ingest all CSVs in a directory (convenience wrapper)."""
    ingestor = CSVIngestor(
        input_dir=input_dir, schema=schema, encoding=encoding,
        fallback_encodings=fallback_encodings,
        on_malformed=on_malformed, logger=logger,
    )
    return ingestor.ingest_directory(
        pattern=pattern, recursive=recursive, combine=combine
    )
