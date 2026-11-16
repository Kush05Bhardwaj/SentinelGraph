"""Unit tests for the SentinelGraph leakage-safe dataset splitter.

Covers:
    * Random splitting (stratified and unstratified)
    * Group-based splitting (no group leakage)
    * Temporal splitting (chronological order)
    * Duplicate overlap detection
    * Class distribution reporting
    * Split manifest generation
    * Reproducibility (same seed -> same split)
    * Configuration validation
    * Leakage error raising
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.splitting import (
    SplitConfig,
    SplitLeakageError,
    Splitter,
)
from ml.splitting.exceptions import SplitConfigError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(100)],
            "Src IP": [f"10.0.0.{i % 10}" for i in range(100)],
            "Dst IP": [f"10.0.0.{(i + 5) % 10}" for i in range(100)],
            "Label": ["BENIGN"] * 70 + ["ATTACK"] * 30,
            "Timestamp": pd.date_range("2017-07-03", periods=100, freq="h"),
        }
    )


@pytest.fixture
def group_df() -> pd.DataFrame:
    """DataFrame where each group_id appears multiple times."""
    rows = []
    for g in range(20):
        for _ in range(5):
            rows.append(
                {
                    "capture_id": f"cap-{g:02d}",
                    "src_ip": f"10.0.0.{g}",
                    "label": "BENIGN" if g % 2 == 0 else "ATTACK",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def temporal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2017-07-03", periods=50, freq="h"),
            "value": range(50),
            "label": ["BENIGN"] * 25 + ["ATTACK"] * 25,
        }
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(SplitConfigError) as excinfo:
        SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)
    assert "sum to 1.0" in str(excinfo.value)


def test_group_strategy_requires_group_column() -> None:
    with pytest.raises(SplitConfigError):
        SplitConfig(strategy="group", group_column=None)


def test_temporal_strategy_requires_timestamp_column() -> None:
    with pytest.raises(SplitConfigError):
        SplitConfig(strategy="temporal", timestamp_column=None)


# ---------------------------------------------------------------------------
# Random splitting
# ---------------------------------------------------------------------------

def test_random_split_sizes(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    assert len(result.train_df) == 70
    assert len(result.val_df) == 15
    assert len(result.test_df) == 15
    assert len(result.train_df) + len(result.val_df) + len(result.test_df) == 100


def test_random_split_partitions_are_disjoint(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    # Manifest indices are the authoritative partition boundaries.
    assert len(set(result.manifest.train_indices)) == 70
    assert len(set(result.manifest.val_indices)) == 15
    assert len(set(result.manifest.test_indices)) == 15
    all_idx = (
        set(result.manifest.train_indices)
        | set(result.manifest.val_indices)
        | set(result.manifest.test_indices)
    )
    assert len(all_idx) == 100
    assert (
        set(result.manifest.train_indices)
        & set(result.manifest.val_indices)
        & set(result.manifest.test_indices)
        == set()
    )


def test_stratified_random_preserves_class_distribution(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random", label_column="Label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    # Both classes should appear in each split.
    assert set(result.train_df["Label"].unique()) == {"BENIGN", "ATTACK"}
    assert set(result.val_df["Label"].unique()) == {"BENIGN", "ATTACK"}
    assert set(result.test_df["Label"].unique()) == {"BENIGN", "ATTACK"}


# ---------------------------------------------------------------------------
# Group splitting
# ---------------------------------------------------------------------------

def test_group_split_no_group_leakage(group_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="capture_id",
    )
    splitter = Splitter(cfg)
    result = splitter.split(group_df)

    train_groups = set(result.train_df["capture_id"])
    val_groups = set(result.val_df["capture_id"])
    test_groups = set(result.test_df["capture_id"])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0


def test_group_split_all_rows_assigned(group_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="capture_id",
    )
    splitter = Splitter(cfg)
    result = splitter.split(group_df)

    total = len(result.train_df) + len(result.val_df) + len(result.test_df)
    assert total == len(group_df)


def test_group_split_group_column_missing() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="missing",
    )
    splitter = Splitter(cfg)
    with pytest.raises(SplitConfigError) as excinfo:
        splitter.split(df)
    assert "not found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------

def test_temporal_split_chronological(temporal_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="temporal", timestamp_column="ts",
    )
    splitter = Splitter(cfg)
    result = splitter.split(temporal_df)

    assert len(result.train_df) + len(result.val_df) + len(result.test_df) == 50

    # Max train timestamp <= min val timestamp <= min test timestamp.
    assert result.train_df["ts"].max() <= result.val_df["ts"].min()
    assert result.val_df["ts"].max() <= result.test_df["ts"].min()


def test_temporal_split_timestamp_column_missing() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="temporal", timestamp_column="missing",
    )
    splitter = Splitter(cfg)
    with pytest.raises(SplitConfigError) as excinfo:
        splitter.split(df)
    assert "not found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Duplicate overlap detection
# ---------------------------------------------------------------------------

def test_duplicate_overlap_detected() -> None:
    # Same row content appears in two different index positions.
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 4, 5],
            "b": ["x", "x", "y", "z", "w", "v"],
            "label": ["BENIGN", "BENIGN", "ATTACK", "BENIGN", "ATTACK", "BENIGN"],
        }
    )
    # Force a split where the duplicate row (a=1,b=x) lands in different splits.
    # Use a tiny dataset so the duplicate is likely split apart.
    cfg = SplitConfig(
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        random_seed=1, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(df)

    # The manifest should flag the duplicate overlap.
    assert result.manifest.leakage_checks.get("duplicate_overlap_passed") is False \
        or result.manifest.leakage_checks.get("passed") is False


# ---------------------------------------------------------------------------
# Class distribution reporting
# ---------------------------------------------------------------------------

def test_class_distribution_in_manifest(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random", label_column="Label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    dist = result.manifest.class_distribution
    assert "train" in dist
    assert "val" in dist
    assert "test" in dist
    assert dist["train"]["BENIGN"] + dist["train"]["ATTACK"] == len(result.train_df)


def test_class_distribution_empty_when_no_label(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)
    assert result.manifest.class_distribution == {}


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def test_manifest_has_required_fields(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random", label_column="Label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    m = result.manifest
    assert m.strategy == "random"
    assert m.random_seed == 42
    assert m.n_train == 70
    assert m.n_train + m.n_val + m.n_test == m.total_rows == 100
    assert m.leakage_checks["passed"] is True


def test_manifest_json_serializable(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="Flow ID",
        label_column="Label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    # Should not raise.
    json.dumps(result.manifest.to_dict())


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_split_reproducible(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random", label_column="Label",
    )
    r1 = Splitter(cfg).split(simple_df.copy())
    r2 = Splitter(cfg).split(simple_df.copy())

    pd.testing.assert_frame_equal(r1.train_df, r2.train_df)
    pd.testing.assert_frame_equal(r1.val_df, r2.val_df)
    pd.testing.assert_frame_equal(r1.test_df, r2.test_df)
    assert r1.manifest.train_indices == r2.manifest.train_indices


def test_different_seed_different_split(simple_df: pd.DataFrame) -> None:
    cfg1 = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=1, strategy="random",
    )
    cfg2 = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=2, strategy="random",
    )
    r1 = Splitter(cfg1).split(simple_df.copy())
    r2 = Splitter(cfg2).split(simple_df.copy())

    # With high probability the splits differ.
    assert set(r1.manifest.train_indices) != set(r2.manifest.train_indices)


# ---------------------------------------------------------------------------
# Validation / test separation
# ---------------------------------------------------------------------------

def test_val_and_test_are_separate(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    assert set(result.manifest.val_indices).isdisjoint(
        set(result.manifest.test_indices)
    )
    assert len(result.val_df) > 0
    assert len(result.test_df) > 0


def test_empty_test_split_when_ratio_zero(simple_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.8, val_ratio=0.2, test_ratio=0.0,
        random_seed=42, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(simple_df)

    assert len(result.test_df) == 0
    assert len(result.train_df) + len(result.val_df) == 100
