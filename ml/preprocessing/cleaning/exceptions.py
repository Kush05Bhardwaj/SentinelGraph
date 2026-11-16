"""Custom exceptions for the SentinelGraph preprocessing/cleaning module."""

from __future__ import annotations


class PreprocessingError(Exception):
    """Base exception for all preprocessing-related errors."""


class PreprocessingNotFittedError(PreprocessingError):
    """Raised when transform is called before fit."""


class PreprocessingSchemaError(PreprocessingError):
    """Raised when input data does not match the expected schema."""


class PreprocessingColumnError(PreprocessingError):
    """Raised when a required column is missing or invalid."""


class PreprocessingArtifactError(PreprocessingError):
    """Raised when preprocessing artifacts cannot be loaded or saved."""
