"""Command-line interface for the SentinelGraph dataset preparation pipeline.

Usage::

    # Local execution
    python -m ml.pipeline.cli \
        --input-dir data/raw/cic-ids2017 \
        --output-dir artifacts/pipeline \
        --strategy group \
        --group-column capture_file \
        --label-column label \
        --random-seed 42

    # Kaggle execution (input from /kaggle/input, output to /kaggle/working)
    python -m ml.pipeline.cli \
        --input-dir /kaggle/input/cic-ids2017 \
        --output-dir /kaggle/working/artifacts/pipeline \
        --strategy group \
        --group-column capture_file \
        --label-label label
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

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
from ml.pipeline.config import IngestionConfig, PipelineConfig
from ml.pipeline.orchestrator import PipelineOrchestrator


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="SentinelGraph dataset preparation pipeline",
    )

    # Input/output
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input CSV files (local or /kaggle/input/...).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for processed artifacts (local or /kaggle/working/...).",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File glob pattern for input CSVs (default: *.csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively for CSV files.",
    )

    # Splitting
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "group", "temporal"],
        help="Split strategy (default: random).",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Group column for group-based splitting (e.g. capture_file).",
    )
    parser.add_argument(
        "--timestamp-column",
        default=None,
        help="Timestamp column for temporal splitting.",
    )
    parser.add_argument(
        "--label-column",
        default="Label",
        help="Target label column name (default: Label).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # Preprocessing
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Primary file encoding (default: utf-8).",
    )
    parser.add_argument(
        "--on-malformed",
        default="skip",
        choices=["skip", "warn", "error"],
        help="Handling for malformed rows (default: skip).",
    )
    parser.add_argument(
        "--scaling",
        default="standard",
        choices=["standard", "minmax", "robust", "none"],
        help="Numerical scaling method (default: standard).",
    )
    parser.add_argument(
        "--encoding-method",
        default="label",
        choices=["label", "onehot", "ordinal", "none"],
        help="Categorical encoding method (default: label).",
    )
    parser.add_argument(
        "--missing-strategy",
        default="impute_median",
        choices=["drop", "impute_mean", "impute_median", "impute_mode", "impute_constant", "none"],
        help="Missing value strategy (default: impute_median).",
    )

    # Feature engineering
    parser.add_argument(
        "--zero-duration-fallback",
        type=float,
        default=1e-6,
        help="Fallback value for zero-duration flows (default: 1e-6).",
    )

    # Pipeline options
    parser.add_argument(
        "--preserve-raw-identifiers",
        action="store_true",
        help="Preserve raw identifier columns in output.",
    )
    parser.add_argument(
        "--no-processed-splits",
        action="store_true",
        help="Skip exporting processed split CSVs.",
    )
    parser.add_argument(
        "--no-quality-report",
        action="store_true",
        help="Skip exporting the dataset quality report.",
    )
    parser.add_argument(
        "--schema-required-columns",
        default=None,
        help="Comma-separated required columns for schema validation.",
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments."""
    required_cols: List[str] = []
    if args.schema_required_columns:
        required_cols = [c.strip() for c in args.schema_required_columns.split(",") if c.strip()]

    return PipelineConfig(
        ingestion=IngestionConfig(
            input_dir=args.input_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            encoding=args.encoding,
            on_malformed=args.on_malformed,
            schema_required_columns=required_cols,
        ),
        preprocessing=PreprocessingConfig(
            label_column=args.label_column,
            missing_values=MissingValueConfig(strategy=args.missing_strategy),
            infinity=InfinityConfig(
                replace_with_nan=True, impute_strategy="median"
            ),
            duplicates=DuplicateConfig(investigate=True, drop_exact=False),
            scaling=ScalingConfig(method=args.scaling),
            encoding=EncodingConfig(method=args.encoding_method),
            normalize_column_names=True,
            preserve_original_labels=True,
            random_seed=args.random_seed,
        ),
        splitting=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            strategy=args.strategy,
            group_column=args.group_column,
            timestamp_column=args.timestamp_column,
            label_column=args.label_column,
        ),
        duration=DurationConfig(zero_duration_fallback=args.zero_duration_fallback),
        feature_engineering=FeatureEngineeringConfig(
            duration=DurationConfig(zero_duration_fallback=args.zero_duration_fallback),
            rate=RateConfig(),
            flags=FlagConfig(),
            port=PortConfig(),
            frequency=FrequencyConfig(),
        ),
        output_dir=args.output_dir,
        preserve_raw_identifiers=args.preserve_raw_identifiers,
        export_processed_splits=not args.no_processed_splits,
        export_quality_report=not args.no_quality_report,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the dataset preparation pipeline.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = build_parser()
    args = parse_args(argv)

    try:
        config = config_from_args(args)
    except Exception as exc:  # pragma: no cover - config validation
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()
    except Exception as exc:  # pragma: no cover - execution depends on input
        print(f"Pipeline execution failed: {exc}", file=sys.stderr)
        return 1

    print("Pipeline completed successfully.")
    print(f"  Train rows: {result.report.train_rows}")
    print(f"  Val rows: {result.report.val_rows}")
    print(f"  Test rows: {result.report.test_rows}")
    print(f"  Artifacts: {len(result.report.artifact_paths)}")
    for path in result.report.artifact_paths:
        print(f"    - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
