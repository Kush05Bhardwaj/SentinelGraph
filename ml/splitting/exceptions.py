"""Custom exceptions for the SentinelGraph dataset splitting module."""

from __future__ import annotations


class SplitError(Exception):
    """Base exception for all splitting-related errors."""


class SplitConfigError(SplitError):
    """Raised when the split configuration is inconsistent or invalid."""


class SplitLeakageError(SplitError):
    """Raised when a leakage check fails (e.g. overlapping groups/duplicates)."""
