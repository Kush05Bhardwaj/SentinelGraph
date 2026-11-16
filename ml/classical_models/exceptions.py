"""Custom exceptions for the SentinelGraph classical ML models package."""

from __future__ import annotations


class ClassicalModelError(Exception):
    """Base exception for classical ML model errors."""


class ModelNotFittedError(ClassicalModelError):
    """Raised when predict is called before fit."""


class ModelConfigurationError(ClassicalModelError):
    """Raised when the model configuration is invalid."""


class ModelSerializationError(ClassicalModelError):
    """Raised when a model cannot be saved or loaded."""


class UnsupportedTaskError(ClassicalModelError):
    """Raised when an unsupported task (binary/multiclass) is requested."""
