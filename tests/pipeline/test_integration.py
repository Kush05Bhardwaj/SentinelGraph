"""Integration tests for the dataset preparation pipeline CLI.

Covers:
    * CLI invocation with local paths
    * CLI invocation with Kaggle-style paths
    * Invalid arguments produce non-zero exit codes
    * End-to-end pipeline via CLI
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from ml.pipeline.cli import main, config_from_args, build_parser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_dir(tmp_path: Path) -> str:
    """Create a temporary directory with sample CSV files."""
    data_dir = tmp_path / "raw" / "cic-ids2017"
    data_dir.mkdir(parents=True)

    monday = pd.DataFrame(
        {
            "Flow ID": [f"m{i}" for i in range(40)],
            "Src IP": [f"10.0.0.{i % 20}" for i in range(40)],
            "Dst IP": [f"10.0.0.{(i + 10) % 20}" for i in range(40)],
            "Src Port": [1000 + i for i in range(40)],
            "Dst Port": [443] * 40,
            "Protocol": ["tcp"] * 40,
            "Flow Duration": [float(i + 1) for i in range(40)],
            "Total Fwd Packets": [10 + i for i in range(40)],
            "Total Backward Packets": [5 + i for i in range(40)],
            "Total Length of Fwd Packets": [1000 + i * 10 for i in range(40)],
            "Total Length of Bwd Packets": [500 + i * 5 for i in range(40)],
            "FIN Flag Count": [0] * 40,
            "SYN Flag Count": [1] * 40,
            "RST Flag Count": [0] * 40,
            "PSH Flag Count": [0] * 40,
            "ACK Flag Count": [1] * 40,
            "URG Flag Count": [0] * 40,
            "Label": ["BENIGN"] * 40,
            "capture_file": ["Monday"] * 40,
        }
    )
    monday.to_csv(data_dir / "Monday-WorkingHours.pcap_ISCX.csv", index=False)

    tuesday = pd.DataFrame(
        {
            "Flow ID": [f"t{i}" for i in range(30)],
            "Src IP": [f"10.0.0.{(i + 30) % 20}" for i in range(30)],
            "Dst IP": [f"10.0.0.{(i + 40) % 20}" for i in range(30)],
            "Src Port": [2000 + i for i in range(30)],
            "Dst Port": [21] * 30,
            "Protocol": ["tcp"] * 30,
            "Flow Duration": [float(i + 1) for i in range(30)],
            "Total Fwd Packets": [20 + i for i in range(30)],
            "Total Backward Packets": [10 + i for i in range(30)],
            "Total Length of Fwd Packets": [2000 + i * 10 for i in range(30)],
            "Total Length of Bwd Packets": [1000 + i * 5 for i in range(30)],
            "FIN Flag Count": [0] * 30,
            "SYN Flag Count": [1] * 30,
            "RST Flag Count": [0] * 30,
            "PSH Flag Count": [0] * 30,
            "ACK Flag Count": [1] * 30,
            "URG Flag Count": [0] * 30,
            "Label": ["FTP-Patator"] * 30,
            "capture_file": ["Tuesday"] * 30,
        }
    )
    tuesday.to_csv(data_dir / "Tuesday-WorkingHours.pcap_ISCX.csv", index=False)

    return str(data_dir)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_end_to_end_local(sample_csv_dir: str) -> None:
    """CLI runs end-to-end with local paths."""
    with tempfile.TemporaryDirectory() as output_dir:
        exit_code = main([
            "--input-dir", sample_csv_dir,
            "--output-dir", output_dir,
            "--strategy", "group",
            "--group-column", "capture_file",
            "--label-column", "Label",
            "--random-seed", "42",
        ])
        assert exit_code == 0

        # Check artifacts exist
        assert os.path.isfile(os.path.join(output_dir, "processed/train.csv"))
        assert os.path.isfile(os.path.join(output_dir, "processed/val.csv"))
        assert os.path.isfile(os.path.join(output_dir, "processed/test.csv"))
        assert os.path.isfile(os.path.join(output_dir, "split_manifest.json"))
        assert os.path.isfile(os.path.join(output_dir, "feature_schema.json"))
        assert os.path.isfile(os.path.join(output_dir, "label_mapping.json"))
        assert os.path.isfile(os.path.join(output_dir, "preprocessing_metadata.json"))
        assert os.path.isfile(os.path.join(output_dir, "data_quality_report.json"))
        assert os.path.isfile(os.path.join(output_dir, "pipeline_report.json"))


def test_cli_kaggle_paths() -> None:
    """CLI accepts Kaggle-style paths (even if files don't exist - tests parsing)."""
    parser = build_parser()
    args = parser.parse_args([
        "--input-dir", "/kaggle/input/cic-ids2017",
        "--output-dir", "/kaggle/working/artifacts/pipeline",
        "--strategy", "random",
        "--label-column", "Label",
    ])
    assert args.input_dir == "/kaggle/input/cic-ids2017"
    assert args.output_dir == "/kaggle/working/artifacts/pipeline"


def test_config_from_args() -> None:
    """config_from_args builds a valid PipelineConfig from CLI args."""
    parser = build_parser()
    args = parser.parse_args([
        "--input-dir", "data/raw/cic-ids2017",
        "--output-dir", "artifacts/pipeline",
        "--strategy", "group",
        "--group-column", "capture_file",
        "--label-column", "Label",
        "--random-seed", "42",
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
    ])
    config = config_from_args(args)

    assert config.ingestion.input_dir == "data/raw/cic-ids2017"
    assert config.output_dir == "artifacts/pipeline"
    assert config.splitting.strategy == "group"
    assert config.splitting.group_column == "capture_file"
    assert config.splitting.label_column == "Label"
    assert config.splitting.train_ratio == 0.6
    assert config.splitting.val_ratio == 0.2
    assert config.splitting.test_ratio == 0.2
    assert config.splitting.random_seed == 42


def test_schema_required_columns_parsed() -> None:
    """--schema-required-columns is parsed correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--input-dir", "data/raw/cic-ids2017",
        "--output-dir", "artifacts/pipeline",
        "--schema-required-columns", "Flow ID,Src IP,Dst IP,Label",
    ])
    config = config_from_args(args)
    assert config.ingestion.schema_required_columns == ["Flow ID", "Src IP", "Dst IP", "Label"]


def test_no_processed_splits_flag(sample_csv_dir: str) -> None:
    """--no-processed-splits skips processed CSV export."""
    with tempfile.TemporaryDirectory() as output_dir:
        exit_code = main([
            "--input-dir", sample_csv_dir,
            "--output-dir", output_dir,
            "--strategy", "random",
            "--label-column", "Label",
            "--no-processed-splits",
        ])
        assert exit_code == 0

        # Processed CSVs should NOT exist
        assert not os.path.isfile(os.path.join(output_dir, "processed/train.csv"))

        # Manifest should still exist
        assert os.path.isfile(os.path.join(output_dir, "split_manifest.json"))


def test_no_quality_report_flag(sample_csv_dir: str) -> None:
    """--no-quality-report skips quality report export."""
    with tempfile.TemporaryDirectory() as output_dir:
        exit_code = main([
            "--input-dir", sample_csv_dir,
            "--output-dir", output_dir,
            "--strategy", "random",
            "--label-column", "Label",
            "--no-quality-report",
        ])
        assert exit_code == 0

        assert not os.path.isfile(os.path.join(output_dir, "data_quality_report.json"))


def test_invalid_input_dir_fails() -> None:
    """CLI fails with non-zero exit when input directory doesn't exist."""
    with tempfile.TemporaryDirectory() as output_dir:
        exit_code = main([
            "--input-dir", "/nonexistent/path/to/data",
            "--output-dir", output_dir,
            "--strategy", "random",
            "--label-column", "Label",
        ])
        assert exit_code == 1


def test_ratios_must_sum_to_one() -> None:
    """CLI rejects split ratios that don't sum to 1.0."""
    parser = build_parser()
    args = parser.parse_args([
        "--input-dir", "data/raw/cic-ids2017",
        "--output-dir", "artifacts/pipeline",
        "--train-ratio", "0.5",
        "--val-ratio", "0.3",
        "--test-ratio", "0.1",
    ])
    # The validation happens in SplitConfig.__post_init__ via config_from_args
    from ml.pipeline.cli import config_from_args
    with pytest.raises(Exception):
        config_from_args(args)
