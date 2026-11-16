"""SentinelGraph dataset ingestion package.

Provides a generic, domain-agnostic CSV ingestor under
``ml/preprocessing/ingestion``:

* :class:`CSVIngestor` — single- and multi-file CSV ingestion.
* :class:`Schema` — declarative schema validation.
* :class:`FileStats` / :class:`IngestionResult` — per-file metrics and output.
* Exceptions in :mod:`sentinelgraph.ml.preprocessing.ingestion.exceptions`.

Usage::

    from sentinelgraph.ml.preprocessing.ingestion import CSVIngestor, Schema

    schema = Schema(required_columns=["src_ip", "dst_ip", "label"])
    ingestor = CSVIngestor(input_dir="data/raw/cic-ids2017", schema=schema)
    results, combined = ingestor.ingest_directory()
"""

from __future__ import annotations

from .exceptions import (
    IngestionEncodingError,
    IngestionEmptyFileError,
    IngestionError,
    IngestionFileNotFoundError,
    IngestionSchemaError,
)
from .ingestor import CSVIngestor, IngestorConfig, ingest_csv, ingest_directory
from .logger import get_logger
from .result import IngestionResult
from .schema import Schema
from .stats import FileStats

__all__ = [
    "CSVIngestor",
    "IngestorConfig",
    "IngestionResult",
    "IngestionError",
    "IngestionFileNotFoundError",
    "IngestionEmptyFileError",
    "IngestionSchemaError",
    "IngestionEncodingError",
    "Schema",
    "FileStats",
    "get_logger",
    "ingest_csv",
    "ingest_directory",
]
