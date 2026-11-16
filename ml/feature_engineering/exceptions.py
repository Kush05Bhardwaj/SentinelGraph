"""Custom exceptions for the SentinelGraph feature engineering module."""

from __future__ import annotations


class FeatureEngineeringError(Exception):
    """Base exception for feature engineering errors."""


class FeatureNotFittedError(FeatureEngineeringError):
    """Raised when transform is called before fit."""


class FeatureSchemaError(FeatureEngineeringError):
    """Raised when input data does not match the expected feature schema."""


class FeatureConfigError(FeatureEngineeringError):
    """Raised when the feature configuration is inconsistent or invalid."""
