"""Custom exceptions for the SentinelGraph dataset preparation pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""


class PipelineConfigError(PipelineError):
    """Raised when the pipeline configuration is inconsistent or invalid."""


class PipelineExecutionError(PipelineError):
    """Raised when a pipeline stage fails during execution."""


class PipelineArtifactError(PipelineError):
    """Raised when pipeline artifacts cannot be loaded or saved."""
