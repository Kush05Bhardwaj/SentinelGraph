"""Integration tests for the preprocessing cleaning pipeline.

Covers leakage safety, fit-on-train-only, artifact persistence/load, label
traceability, and end-to-end train/test preprocessing.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.cleaning import (
    ColumnConfig,
    PreprocessingConfig,
    PreprocessingTransformer,
    ScalingConfig,
    EncodingConfig,
    MissingValueConfig,
    InfinityConfig,
    DuplicateConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 200
    return pd.DataFrame(
        {
            "src_ip": [f"10.0.0.{i % 250}" for i in range(n)],
            "dst_ip": [f"10.0.0.{ (i + 50) % 250}" for i in range(n)],
            "bytes": np.random.exponential(1000, size=n).round(2),
            "duration": np.random.exponential(5, size=n).round(3),
            "protocol": np.random.choice(["tcp", "udp", "icmp"], size=n),
            "flag": np.random.choice(["SYN", "ACK", "FIN"], size=n),
            "label": np.random.choice(["BENIGN", "ATTACK"], size=n, p=[0.8, 0.2]),
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    np.random.seed(1)
    n = 50
    return pd.DataFrame(
        {
            "src_ip": [f"10.0.0.{i % 250}" for i in range(n)],
            "dst_ip": [f"10.0.0.{ (i + 50) % 250}" for i in range(n)],
            "bytes": np.random.exponential(1000, size=n).round(2),
            "duration": np.random.exponential(5, size=n).round(3),
            "protocol": np.random.choice(["tcp", "udp", "icmp"], size=n),
            "flag": np.random.choice(["SYN", "ACK", "FIN"], size=n),
            "label": np.random.choice(["BENIGN", "ATTACK"], size=n, p=[0.8, 0.2]),
        }
    )


# ---------------------------------------------------------------------------
# Leakage safety: fit on train only
# ---------------------------------------------------------------------------

def test_scaling_params_come_from_train_only(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        identifier_columns=["src_ip", "dst_ip"],
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
        preserve_original_labels=True,
    )
    pre = PreprocessingTransformer(cfg)
    train_res = pre.fit_transform(train_df.copy())
    test_res = pre.transform(test_df.copy())

    # Train features should be standardized to ~mean 0, ~std 1.
    for col in ["bytes", "duration"]:
        assert abs(train_res.X[col].mean()) < 1e-6
        assert abs(train_res.X[col].std(ddof=0) - 1.0) < 1e-6

    # Test features should use TRAIN mean/scale (not test stats), so they will
    # NOT be mean 0 / std 1 generally.
    test_mean = test_res.X["bytes"].mean()
    assert abs(test_mean) > 1e-6, "Test bytes should not be re-centered (leakage!)"

    # Train mean/scale recorded in scaler_params.
    assert "bytes" in pre._scaler_params
    assert pre._scaler_params["bytes"]["mean"] == pytest.approx(
        train_df["bytes"].mean(), rel=1e-6
    )


def test_encoder_learned_from_train_only(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        identifier_columns=["src_ip", "dst_ip"],
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="label"),
    )
    pre = PreprocessingTransformer(cfg)
    pre.fit(train_df.copy())

    # Encoder categories come from train.
    assert "protocol" in pre._encoder_mappings
    train_cats = set(train_df["protocol"].unique())
    assert set(pre._encoder_mappings["protocol"].keys()) == train_cats

    # Transforming test with an unseen category handled gracefully.
    unseen = test_df.copy()
    unseen["protocol"] = ["ipv6"] * len(unseen)
    with pytest.raises(Exception):
        pre.transform(unseen)


# ---------------------------------------------------------------------------
# No silent column drops / report completeness
# ---------------------------------------------------------------------------

def test_no_silent_column_drop(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        identifier_columns=["src_ip", "dst_ip"],
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df.copy())
    # All non-identifier, non-label columns should be preserved as features.
    expected = {"bytes", "duration", "protocol", "flag"}
    assert expected.issubset(set(result.X.columns))


def test_report_documents_transformations(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
        missing_values=MissingValueConfig(strategy="impute_median"),
        infinity=InfinityConfig(replace_with_nan=True, impute_strategy="median"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df.copy())

    assert result.report.fitted
    assert result.report.n_columns_input == train_df.shape[1]
    assert len(result.report.transformations) > 0
    # Every column should have a report entry.
    reported = {cr.name for cr in result.report.column_reports}
    assert set(train_df.columns).issubset(reported)


# ---------------------------------------------------------------------------
# Label traceability
# ---------------------------------------------------------------------------

def test_original_labels_preserved(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
        preserve_original_labels=True,
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df.copy())
    assert result.y_raw is not None
    pd.testing.assert_series_equal(
        result.y_raw.reset_index(drop=True), train_df["label"].reset_index(drop=True)
    )


def test_label_column_not_in_features(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df.copy())
    assert "label" not in result.X.columns


# ---------------------------------------------------------------------------
# Artifact persistence / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_artifacts_roundtrip(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        identifier_columns=["src_ip", "dst_ip"],
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
        missing_values=MissingValueConfig(strategy="impute_median"),
    )
    pre = PreprocessingTransformer(cfg)
    train_res = pre.fit_transform(train_df.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = os.path.join(tmpdir, "preprocessing")
        pre.save_artifacts(artifact_dir)

        # Files exist.
        assert os.path.isfile(os.path.join(artifact_dir, "preprocessing_report.json"))
        assert os.path.isfile(os.path.join(artifact_dir, "fitted_state.json"))

        # Load into a new transformer.
        loaded = PreprocessingTransformer.load_artifacts(artifact_dir)
        assert loaded._fitted
        assert loaded._scaler_params == pre._scaler_params
        assert loaded._encoder_mappings == pre._encoder_mappings
        assert loaded._feature_columns == pre._feature_columns

        # Loaded transformer produces identical transform output.
        loaded_res = loaded.transform(test_df.copy())
        pd.testing.assert_frame_equal(train_res.X.iloc[:0], loaded_res.X.iloc[:0])  # schema
        # Full equality on test transform.
        pd.testing.assert_frame_equal(
            pre.transform(test_df.copy()).X, loaded_res.X
        )


def test_report_json_serializable(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
    )
    pre = PreprocessingTransformer(cfg)
    pre.fit_transform(train_df.copy())

    report_dict = pre._report.to_dict()
    # Should not raise.
    json.dumps(report_dict)


# ---------------------------------------------------------------------------
# End-to-end train -> test pipeline
# ---------------------------------------------------------------------------

def test_train_test_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="label",
        identifier_columns=["src_ip", "dst_ip"],
        missing_values=MissingValueConfig(strategy="impute_median"),
        infinity=InfinityConfig(replace_with_nan=True, impute_strategy="median"),
        duplicates=DuplicateConfig(investigate=True, drop_exact=False),
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
        preserve_original_labels=True,
    )
    pre = PreprocessingTransformer(cfg)

    train_res = pre.fit_transform(train_df.copy())
    test_res = pre.transform(test_df.copy())

    # No NaNs in output (imputed).
    assert not train_res.X.isna().any().any()
    assert not test_res.X.isna().any().any()

    # No infinite values.
    assert not np.isinf(train_res.X.select_dtypes(include=[np.number])).any().any()
    assert not np.isinf(test_res.X.select_dtypes(include=[np.number])).any().any()

    # Labels present and aligned.
    assert train_res.y is not None
    assert len(train_res.y) == len(train_res.X)
    assert len(test_res.y) == len(test_res.X)

    # Original labels preserved.
    assert train_res.y_raw is not None
    assert len(train_res.y_raw) == len(train_res.X)

    # Report documents the full pipeline.
    assert pre._report.fitted
    assert len(pre._report.transformations) > 0
