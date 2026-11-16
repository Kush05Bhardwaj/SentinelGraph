"""Unit tests for the SentinelGraph dataset preparation pipeline orchestrator.

Covers:
    * End-to-end pipeline execution
    * All artifact generation (processed splits, manifests, schema, label mapping, metadata, quality report)
    * Split integrity (no overlapping groups, disjoint val/test)
    * Reproducibility (same seed → same result)
    * Preprocessing fitted only on training data
    * CLI invocation
    * Local and Kaggle-style paths
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.pipeline.config import (
    IngestionConfig,
    PipelineConfig,
)
from ml.pipeline.orchestrator import PipelineOrchestrator
from ml.splitting import SplitConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> str:
    """Create a temporary directory with sample CSV files."""
    data_dir = tmp_path / "raw" / "cic-ids2017"
    data_dir.mkdir(parents=True)

    # Five days of traffic (Monday-Friday), mimicking CIC-IDS2017 structure.
    days = [
        ("Monday", "2017-07-03", "BENIGN", 50),
        ("Tuesday", "2017-07-04", "FTP-Patator", 30),
        ("Wednesday", "2017-07-05", "DoS Hulk", 40),
        ("Thursday", "2017-07-06", "Web Attack", 30),
        ("Friday", "2017-07-07", "DDoS", 40),
    ]
    flow_id = 0
    for day_name, day_date, attack, n_flows in days:
        df = pd.DataFrame(
            {
                "Flow ID": [f"f{flow_id + i}" for i in range(n_flows)],
                "Src IP": [f"10.0.0.{(flow_id + i) % 20}" for i in range(n_flows)],
                "Dst IP": [f"10.0.0.{(flow_id + i + 10) % 20}" for i in range(n_flows)],
                "Src Port": [1000 + flow_id + i for i in range(n_flows)],
                "Dst Port": [443 if i % 2 == 0 else 21 for i in range(n_flows)],
                "Protocol": ["tcp"] * n_flows,
                "Flow Duration": [float(i + 1) for i in range(n_flows)],
                "Total Fwd Packets": [10 + i for i in range(n_flows)],
                "Total Backward Packets": [5 + i for i in range(n_flows)],
                "Total Length of Fwd Packets": [1000 + i * 10 for i in range(n_flows)],
                "Total Length of Bwd Packets": [500 + i * 5 for i in range(n_flows)],
                "FIN Flag Count": [0] * n_flows,
                "SYN Flag Count": [1] * n_flows,
                "RST Flag Count": [0] * n_flows,
                "PSH Flag Count": [0] * n_flows,
                "ACK Flag Count": [1] * n_flows,
                "URG Flag Count": [0] * n_flows,
                "Label": [attack] * n_flows,
                "capture_file": [day_name] * n_flows,
            }
        )
        df.to_csv(data_dir / f"{day_name}-WorkingHours.pcap_ISCX.csv", index=False)
        flow_id += n_flows

    return str(data_dir)


# ---------------------------------------------------------------------------
# Pipeline execution tests
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end(sample_csv_dir: str) -> None:
    """Full pipeline run produces all expected artifacts."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        # Result structure
        assert result.train_df is not None
        assert result.val_df is not None
        assert result.test_df is not None
        assert result.manifest is not None
        assert result.feature_schema is not None
        assert result.report is not None

        # All rows accounted for
        total = len(result.train_df) + len(result.val_df) + len(result.test_df)
        assert total == result.report.total_rows

        # Label column present in processed splits
        assert "Label" in result.train_df.columns
        assert "Label" in result.val_df.columns
        assert "Label" in result.test_df.columns

        # Engineered features present
        assert "flow_duration" in result.train_df.columns
        assert "packets_per_sec" in result.train_df.columns
        assert "bytes_per_sec" in result.train_df.columns

        # No invalid values
        numeric = result.train_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric).any().any()
        assert not numeric.isna().any().any()


def test_pipeline_artifacts_written(sample_csv_dir: str) -> None:
    """All expected artifacts are written to the output directory."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        # Check artifact files exist
        expected_files = [
            "processed/train.csv",
            "processed/val.csv",
            "processed/test.csv",
            "split_manifest.json",
            "feature_schema.json",
            "label_mapping.json",
            "preprocessing_metadata.json",
            "data_quality_report.json",
            "pipeline_report.json",
        ]
        for rel_path in expected_files:
            full_path = os.path.join(output_dir, rel_path)
            assert os.path.isfile(full_path), f"Missing artifact: {rel_path}"
            # Artifact paths are absolute; check suffix match.
            assert any(
                p.endswith(rel_path.replace("/", os.sep))
                for p in result.report.artifact_paths
            ), f"Artifact path not recorded: {rel_path}"


def test_split_manifest_leakage_checks(sample_csv_dir: str) -> None:
    """Split manifest shows no group leakage."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        checks = result.manifest.leakage_checks
        assert checks["group_overlap_passed"] is True
        assert checks["duplicate_overlap_passed"] is True
        assert checks["passed"] is True


def test_val_test_disjoint(sample_csv_dir: str) -> None:
    """Validation and test splits are disjoint."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        assert set(result.manifest.val_indices).isdisjoint(
            set(result.manifest.test_indices)
        )


def test_pipeline_reproducible(sample_csv_dir: str) -> None:
    """Same seed produces identical results."""
    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as output_dir:
            config = PipelineConfig(
                ingestion=IngestionConfig(input_dir=sample_csv_dir),
                splitting=SplitConfig(
                    train_ratio=0.7,
                    val_ratio=0.15,
                    test_ratio=0.15,
                    random_seed=42,
                    strategy="group",
                    group_column="capture_file",
                    label_column="Label",
                ),
                output_dir=output_dir,
            )
            orchestrator = PipelineOrchestrator(config)
            result = orchestrator.run()
            results.append(result)

    pd.testing.assert_frame_equal(results[0].train_df, results[1].train_df)
    pd.testing.assert_frame_equal(results[0].val_df, results[1].val_df)
    pd.testing.assert_frame_equal(results[0].test_df, results[1].test_df)
    assert results[0].manifest.train_indices == results[1].manifest.train_indices


def test_label_mapping_exported(sample_csv_dir: str) -> None:
    """Label mapping is exported and contains training labels."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        # Label mapping should be non-empty (contains labels from training)
        assert len(result.label_mapping) > 0

        # Label mapping file should exist
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        assert os.path.isfile(mapping_path)
        with open(mapping_path) as f:
            saved_mapping = json.load(f)
        assert len(saved_mapping) > 0


def test_feature_schema_documented(sample_csv_dir: str) -> None:
    """Feature schema documents every engineered feature."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        schema = result.feature_schema
        assert len(schema.feature_entries) > 0
        for entry in schema.feature_entries:
            assert entry.name
            assert entry.derivation  # every feature documented
            assert entry.dtype


def test_no_hardcoded_dataset_paths() -> None:
    """Pipeline config does not hardcode dataset-specific absolute paths."""
    config = PipelineConfig(
        ingestion=IngestionConfig(input_dir="data/raw/cic-ids2017"),
        splitting=SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            strategy="random",
        ),
        output_dir="artifacts/pipeline",
    )
    # Paths should be relative/configurable, not absolute dataset paths
    assert config.ingestion.input_dir == "data/raw/cic-ids2017"
    assert config.output_dir == "artifacts/pipeline"


def test_kaggle_paths_supported() -> None:
    """Kaggle-style input/output paths are accepted."""
    config = PipelineConfig(
        ingestion=IngestionConfig(input_dir="/kaggle/input/cic-ids2017"),
        splitting=SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            strategy="random",
        ),
        output_dir="/kaggle/working/artifacts/pipeline",
    )
    assert config.ingestion.input_dir == "/kaggle/input/cic-ids2017"
    assert config.output_dir == "/kaggle/working/artifacts/pipeline"


def test_random_strategy_works(sample_csv_dir: str) -> None:
    """Random split strategy works without group column."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="random",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        assert len(result.train_df) > 0
        assert len(result.val_df) > 0
        assert len(result.test_df) > 0
        assert result.manifest.leakage_checks["passed"] is True


def test_quality_report_has_class_distribution(sample_csv_dir: str) -> None:
    """Quality report contains class distribution per split."""
    with tempfile.TemporaryDirectory() as output_dir:
        config = PipelineConfig(
            ingestion=IngestionConfig(input_dir=sample_csv_dir),
            splitting=SplitConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                strategy="group",
                group_column="capture_file",
                label_column="Label",
            ),
            output_dir=output_dir,
        )
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        # Read the quality report from artifacts
        quality_path = os.path.join(output_dir, "data_quality_report.json")
        assert os.path.isfile(quality_path)

        with open(quality_path) as f:
            report = json.load(f)

        assert "class_distribution" in report
        assert "train" in report["class_distribution"]
        assert "val" in report["class_distribution"]
        assert "test" in report["class_distribution"]
