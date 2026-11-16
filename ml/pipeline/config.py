"""Configuration for the dataset preparation pipeline.

All settings are explicit and serializable so the full preparation pipeline
(ingest → clean → feature-engineer → split → export) is deterministic and
reproducible across local and Kaggle executions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ml.feature_engineering import (
    DurationConfig,
    FeatureEngineeringConfig,
    FlagConfig,
    FrequencyConfig,
    PortConfig,
    RateConfig,
)
from ml.preprocessing.cleaning import (
    ColumnConfig,
    DuplicateConfig,
    EncodingConfig,
    InfinityConfig,
    MissingValueConfig,
    PreprocessingConfig,
    ScalingConfig,
)
from ml.splitting import SplitConfig


@dataclass
class IngestionConfig:
    """Configuration for the dataset ingestion stage."""

    input_dir: str
    pattern: str = "*.csv"
    recursive: bool = False
    encoding: str = "utf-8"
    fallback_encodings: tuple = ("utf-8-sig", "latin-1", "cp1252")
    on_malformed: str = "skip"
    schema_required_columns: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Top-level dataset preparation configuration.

    All fields are optional with sensible defaults. The pipeline is fully
    configurable via this dataclass or through CLI arguments.
    """

    ingestion: IngestionConfig = field(default_factory=lambda: IngestionConfig(input_dir=""))
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )
    splitting: SplitConfig = field(default_factory=SplitConfig)
    duration: DurationConfig = field(default_factory=DurationConfig)
    rate: RateConfig = field(default_factory=RateConfig)
    flags: FlagConfig = field(default_factory=FlagConfig)
    port: PortConfig = field(default_factory=PortConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)

    output_dir: str = "artifacts/pipeline"
    preserve_raw_identifiers: bool = False
    export_processed_splits: bool = True
    export_quality_report: bool = True

    def __post_init__(self) -> None:
        # Ensure the feature engineering config reflects sub-configs.
        self.feature_engineering.duration = self.duration
        self.feature_engineering.rate = self.rate
        self.feature_engineering.flags = self.flags
        self.feature_engineering.port = self.port
        self.feature_engineering.frequency = self.frequency

        # Sync label column between splitting and preprocessing configs.
        if (
            self.splitting.label_column
            and not self.preprocessing.label_column
        ):
            self.preprocessing.label_column = self.splitting.label_column

        # Use "ignore" for unknown labels so group-based splits with
        # homogeneous labels per group do not fail at transform time.
        if self.preprocessing.encoding.handle_unknown == "error":
            self.preprocessing.encoding.handle_unknown = "ignore"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingestion": {
                "input_dir": self.ingestion.input_dir,
                "pattern": self.ingestion.pattern,
                "recursive": self.ingestion.recursive,
                "encoding": self.ingestion.encoding,
                "on_malformed": self.ingestion.on_malformed,
                "schema_required_columns": self.ingestion.schema_required_columns,
            },
            "preprocessing": {
                "label_column": self.preprocessing.label_column,
                "identifier_columns": self.preprocessing.identifier_columns,
                "missing_values": {
                    "strategy": self.preprocessing.missing_values.strategy,
                    "threshold": self.preprocessing.missing_values.threshold,
                },
                "infinity": {
                    "replace_with_nan": self.preprocessing.infinity.replace_with_nan,
                    "impute_strategy": self.preprocessing.infinity.impute_strategy,
                },
                "duplicates": {
                    "investigate": self.preprocessing.duplicates.investigate,
                    "drop_exact": self.preprocessing.duplicates.drop_exact,
                },
                "scaling": {
                    "method": self.preprocessing.scaling.method,
                    "with_mean": self.preprocessing.scaling.with_mean,
                    "with_std": self.preprocessing.scaling.with_std,
                },
                "encoding": {
                    "method": self.preprocessing.encoding.method,
                    "handle_unknown": self.preprocessing.encoding.handle_unknown,
                },
                "normalize_column_names": self.preprocessing.normalize_column_names,
                "preserve_original_labels": self.preprocessing.preserve_original_labels,
                "random_seed": self.preprocessing.random_seed,
            },
            "splitting": {
                "train_ratio": self.splitting.train_ratio,
                "val_ratio": self.splitting.val_ratio,
                "test_ratio": self.splitting.test_ratio,
                "random_seed": self.splitting.random_seed,
                "strategy": self.splitting.strategy,
                "group_column": self.splitting.group_column,
                "timestamp_column": self.splitting.timestamp_column,
                "label_column": self.splitting.label_column,
            },
            "output_dir": self.output_dir,
            "preserve_raw_identifiers": self.preserve_raw_identifiers,
            "export_processed_splits": self.export_processed_splits,
            "export_quality_report": self.export_quality_report,
        }
