"""Pipeline report: documents the full dataset preparation run.

The report records stage timings, row counts at each stage, artifact paths,
and any warnings produced during the pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageReport:
    """Report for a single pipeline stage."""

    name: str
    status: str = "pending"  # pending | completed | failed
    rows_in: int = 0
    rows_out: int = 0
    duration_seconds: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class PipelineReport:
    """Full report of a pipeline run."""

    pipeline_version: str = "1.0"
    config: Dict[str, Any] = field(default_factory=dict)
    stages: List[StageReport] = field(default_factory=list)
    artifact_paths: List[str] = field(default_factory=list)
    total_rows: int = 0
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def add_stage(self, stage: StageReport) -> None:
        self.stages.append(stage)

    def add_artifact(self, path: str) -> None:
        self.artifact_paths.append(path)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "config": self.config,
            "stages": [s.to_dict() for s in self.stages],
            "artifact_paths": self.artifact_paths,
            "total_rows": self.total_rows,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "test_rows": self.test_rows,
            "warnings": self.warnings,
            "error": self.error,
        }
