"""Public API for the pipeline package."""

from __future__ import annotations

from .cli import main
from .config import IngestionConfig, PipelineConfig
from .exceptions import (
    PipelineArtifactError,
    PipelineConfigError,
    PipelineError,
    PipelineExecutionError,
)
from .orchestrator import PipelineOrchestrator, PipelineResult
from .report import PipelineReport, StageReport

__all__ = [
    "PipelineOrchestrator",
    "PipelineResult",
    "PipelineConfig",
    "IngestionConfig",
    "PipelineReport",
    "StageReport",
    "PipelineError",
    "PipelineConfigError",
    "PipelineExecutionError",
    "PipelineArtifactError",
    "main",
]
