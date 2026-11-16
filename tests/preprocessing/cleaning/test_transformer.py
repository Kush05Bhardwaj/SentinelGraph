"""Unit tests for the preprocessing cleaning transformer.

Covers:
    * Column normalization (whitespace, lowercase)
    * Data type correction
    * Missing value handling (all strategies)
    * Infinity handling
    * Duplicate investigation
    * Invalid numerical values
    * Categorical normalization
    * Label normalization
    * Numerical scaling (standard, minmax, robust)
    * Categorical encoding (label, onehot, ordinal)
    * Deterministic behavior
    * Not-fitted error
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.cleaning import (
    ColumnConfig,
    DuplicateConfig,
    EncodingConfig,
    InfinityConfig,
    MissingValueConfig,
    PreprocessingConfig,
    PreprocessingNotFittedError,
    PreprocessingSchemaError,
    PreprocessingTransformer,
    ScalingConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Src IP": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.1"],
            "Dst IP": ["10.0.0.2", "10.0.0.3", "10.0.0.1", "10.0.0.4"],
            "Bytes": [100.0, 200.0, 300.0, 400.0],
            "Duration": [1.0, 2.0, 3.0, 4.0],
            "Protocol": ["tcp", "udp", "tcp", "icmp"],
            "Label": ["BENIGN", "ATTACK", "BENIGN", "ATTACK"],
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Src IP": ["10.0.0.5", "10.0.0.6"],
            "Dst IP": ["10.0.0.1", "10.0.0.2"],
            "Bytes": [150.0, 250.0],
            "Duration": [1.5, 2.5],
            "Protocol": ["tcp", "udp"],
            "Label": ["BENIGN", "ATTACK"],
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " Src IP ": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.1"],
            "Bytes": [100.0, 200.0, np.nan, 400.0],
            "Duration": [1.0, np.inf, 3.0, 4.0],
            "Protocol": ["tcp", "udp", "tcp", "icmp"],
            "Label ": ["BENIGN", "ATTACK", "BENIGN", "ATTACK"],
        }
    )


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def test_column_name_whitespace_stripped(train_df: pd.DataFrame) -> None:
    df = train_df.copy()
    df.columns = [" Src IP ", "Dst IP", " Bytes ", "Duration", "Protocol", "Label"]
    cfg = PreprocessingConfig(
        normalize_column_names=True, label_column="Label"
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(df)
    assert "Src IP" in result.X.columns
    assert "Bytes" in result.X.columns
    assert "Src IP " not in result.X.columns


def test_column_names_lowercased(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        normalize_column_names=True,
        lowercase_column_names=True,
        label_column="label",
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df.rename(columns={"Label": "label"}))
    assert "label" not in result.X.columns  # label separated into y
    assert all(c == c.lower() for c in result.X.columns)


# ---------------------------------------------------------------------------
# Data type correction
# ---------------------------------------------------------------------------

def test_column_type_correction(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        columns=[
            ColumnConfig(name="Bytes", dtype="float64"),
            ColumnConfig(name="Duration", dtype="float64"),
        ],
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    train_df["Bytes"] = train_df["Bytes"].astype("object")
    result = pre.fit_transform(train_df)
    assert pd.api.types.is_numeric_dtype(result.X["Bytes"])


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

def test_missing_impute_median(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        missing_values=MissingValueConfig(strategy="impute_median"),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
        preserve_original_labels=True,
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert result.X["Bytes"].isna().sum() == 0
    assert result.X["Duration"].isna().sum() == 0


def test_missing_impute_constant(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        missing_values=MissingValueConfig(
            strategy="impute_constant", constant_value=-1.0
        ),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert (result.X["Bytes"] == -1.0).any()


def test_missing_drop(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        missing_values=MissingValueConfig(strategy="drop"),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert result.X["Bytes"].isna().sum() == 0
    assert len(result.X) <= len(messy_df)


def test_missing_none_leaves_nans(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        missing_values=MissingValueConfig(strategy="none"),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert result.X["Bytes"].isna().sum() > 0


# ---------------------------------------------------------------------------
# Infinity handling
# ---------------------------------------------------------------------------

def test_infinity_impute_median(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        infinity=InfinityConfig(replace_with_nan=True, impute_strategy="median"),
        missing_values=MissingValueConfig(strategy="none"),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert not np.isinf(result.X["Duration"]).any()
    assert result.X["Duration"].notna().all()


def test_infinity_no_replace_keeps_inf(messy_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label ",
        infinity=InfinityConfig(replace_with_nan=False),
        missing_values=MissingValueConfig(strategy="none"),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(messy_df)
    assert np.isinf(result.X["Duration"]).any()


# ---------------------------------------------------------------------------
# Duplicate investigation
# ---------------------------------------------------------------------------

def test_duplicates_investigated_not_dropped(train_df: pd.DataFrame) -> None:
    df = pd.concat([train_df, train_df.iloc[:1]], ignore_index=True)
    cfg = PreprocessingConfig(
        label_column="Label",
        duplicates=DuplicateConfig(investigate=True, drop_exact=False),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(df)
    assert len(result.X) == len(df)  # none dropped
    assert any("duplicate" in t.lower() for t in result.report.transformations)


def test_duplicates_dropped(train_df: pd.DataFrame) -> None:
    df = pd.concat([train_df, train_df.iloc[:1]], ignore_index=True)
    cfg = PreprocessingConfig(
        label_column="Label",
        duplicates=DuplicateConfig(investigate=True, drop_exact=True),
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(df)
    assert len(result.X) == len(train_df)  # duplicate removed


# ---------------------------------------------------------------------------
# Categorical normalization + encoding
# ---------------------------------------------------------------------------

def test_categorical_label_encoding(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        encoding=EncodingConfig(method="label"),
        scaling=ScalingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    assert "Protocol" in result.X.columns
    assert pd.api.types.is_numeric_dtype(result.X["Protocol"])
    assert set(result.X["Protocol"].dropna().unique()).issubset({0, 1, 2})


def test_categorical_onehot_encoding(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        encoding=EncodingConfig(method="onehot"),
        scaling=ScalingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    proto_cols = [c for c in result.X.columns if c.startswith("Protocol__")]
    assert len(proto_cols) == 3  # tcp, udp, icmp
    assert "Protocol" not in result.X.columns


def test_categorical_ordinal_encoding(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        encoding=EncodingConfig(method="ordinal"),
        scaling=ScalingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    assert pd.api.types.is_numeric_dtype(result.X["Protocol"])


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def test_label_normalization(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
        preserve_original_labels=True,
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    assert result.y is not None
    assert set(result.y.unique()).issubset({"BENIGN", "ATTACK"})
    assert result.y_raw is not None
    assert list(result.y_raw) == list(train_df["Label"])


def test_label_column_removed_from_features(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    assert "Label" not in result.X.columns


# ---------------------------------------------------------------------------
# Numerical scaling
# ---------------------------------------------------------------------------

def test_standard_scaling(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    for col in ["Bytes", "Duration"]:
        vals = result.X[col].dropna()
        assert abs(vals.mean()) < 1e-6
        assert abs(vals.std(ddof=0) - 1.0) < 1e-6


def test_minmax_scaling(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="minmax"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    for col in ["Bytes", "Duration"]:
        vals = result.X[col].dropna()
        assert vals.min() >= 0.0 - 1e-6
        assert vals.max() <= 1.0 + 1e-6


def test_robust_scaling(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="robust"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    assert "Bytes" in result.X.columns


def test_no_scaling_leaves_values(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="none"),
        encoding=EncodingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    result = pre.fit_transform(train_df)
    pd.testing.assert_series_equal(result.X["Bytes"], train_df["Bytes"].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises(train_df: pd.DataFrame) -> None:
    pre = PreprocessingTransformer(PreprocessingConfig(label_column="Label"))
    with pytest.raises(PreprocessingNotFittedError):
        pre.transform(train_df)


def test_unseen_label_raises(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        encoding=EncodingConfig(handle_unknown="error"),
        scaling=ScalingConfig(method="none"),
    )
    pre = PreprocessingTransformer(cfg)
    pre.fit(train_df)
    weird = test_df.copy()
    weird["Label"] = ["NEW_LABEL", "NEW_LABEL"]
    with pytest.raises(PreprocessingSchemaError):
        pre.transform(weird)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_fit_transform_deterministic(train_df: pd.DataFrame) -> None:
    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
    )
    r1 = PreprocessingTransformer(cfg).fit_transform(train_df.copy())
    r2 = PreprocessingTransformer(cfg).fit_transform(train_df.copy())
    pd.testing.assert_frame_equal(r1.X, r2.X)
    pd.testing.assert_series_equal(r1.y, r2.y)
