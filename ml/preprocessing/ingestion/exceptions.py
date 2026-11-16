"""Custom exceptions for the SentinelGraph ingestion module."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""


class IngestionFileNotFoundError(IngestionError):
    """Raised when an input file cannot be found."""


class IngestionEmptyFileError(IngestionError):
    """Raised when an input file is empty (no readable content)."""


class IngestionSchemaError(IngestionError):
    """Raised when a dataframe fails schema validation."""


class IngestionEncodingError(IngestionError):
    """Raised when none of the candidate encodings can decode a file."""


class IngestionConfigurationError(IngestionError):
    """Raised when the ingestor is configured inconsistently."""
