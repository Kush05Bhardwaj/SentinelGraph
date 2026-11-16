"""Integration tests for the leakage-safe dataset splitter.

Covers CIC-IDS2017-style grouping (capture file / day), group-based splitting
with temporal ordering, no prohibited duplicate overlap, reproducibility, and
validation/test separation. Also documents unavoidable limitations.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.splitting import SplitConfig, Splitter


# ---------------------------------------------------------------------------
# Fixtures: CIC-IDS2017-style data
# ---------------------------------------------------------------------------

@pytest.fixture
def cic_ids2017_style_df() -> pd.DataFrame:
    """Simulates CIC-IDS2017: per-day capture files with flows.

    Each day (Monday..Friday) is a capture file. Flows within a day share the
    capture file id. Attack type varies by day.
    """
    days = [
        ("Monday", "2017-07-03", "BENIGN"),
        ("Tuesday", "2017-07-04", "FTP-Patator"),
        ("Wednesday", "2017-07-05", "DoS Hulk"),
        ("Thursday", "2017-07-06", "Web Attack"),
        ("Friday", "2017-07-07", "DDoS"),
    ]
    rows = []
    flow_id = 0
    for day_name, day_date, attack in days:
        n_flows = 200 if day_name == "Monday" else 150
        for _ in range(n_flows):
            rows.append(
                {
                    "Flow ID": f"f{flow_id}",
                    "capture_file": day_name,
                    "capture_date": day_date,
                    "src_ip": f"10.0.0.{flow_id % 50}",
                    "dst_ip": f"10.0.0.{(flow_id + 10) % 50}",
                    "timestamp": pd.Timestamp(day_date)
                    + pd.Timedelta(hours=flow_id % 24),
                    "bytes": float(flow_id * 10),
                    "label": "BENIGN" if day_name == "Monday" else attack,
                }
            )
            flow_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CIC-IDS2017 grouping investigation
# ---------------------------------------------------------------------------

def test_cic_ids2017_grouping_by_capture_file(cic_ids2017_style_df: pd.DataFrame) -> None:
    """CIC-IDS2017 flows should be grouped by capture file (day) to avoid
    leakage: flows from the same capture session share network context."""
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="capture_file",
        label_column="label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(cic_ids2017_style_df)

    # No capture file appears in more than one split.
    train_files = set(result.train_df["capture_file"])
    val_files = set(result.val_df["capture_file"])
    test_files = set(result.test_df["capture_file"])

    assert len(train_files & val_files) == 0
    assert len(train_files & test_files) == 0
    assert len(val_files & test_files) == 0

    # All 5 capture files are assigned.
    assert train_files | val_files | test_files == {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    }


def test_cic_ids2017_temporal_split(cic_ids2017_style_df: pd.DataFrame) -> None:
    """Temporal split respects capture dates: train on earliest days, test on
    latest days (simulating deployment on future traffic)."""
    cfg = SplitConfig(
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_seed=42, strategy="temporal", timestamp_column="timestamp",
        label_column="label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(cic_ids2017_style_df)

    # Train should contain the earliest days (Monday/Tuesday mostly).
    assert result.train_df["capture_date"].min() <= result.val_df["capture_date"].min()
    assert result.val_df["capture_date"].max() <= result.test_df["capture_date"].max()


# ---------------------------------------------------------------------------
# Duplicate overlap
# ---------------------------------------------------------------------------

def test_no_prohibited_duplicate_overlap(cic_ids2017_style_df: pd.DataFrame) -> None:
    """Exact duplicate row content must not span splits."""
    # Inject a duplicate row into two different logical groups to force the
    # checker to exercise the duplicate-overlap path.
    df = cic_ids2017_style_df.copy()
    duplicate_row = df.iloc[0].copy()
    df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)

    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=7, strategy="random",
    )
    splitter = Splitter(cfg)
    result = splitter.split(df)

    checks = result.manifest.leakage_checks
    # Either no overlap (passed) or overlap is flagged, never silently ignored.
    assert checks.get("duplicate_overlap_passed") is True \
        or (checks.get("passed") is False
            and "Duplicate rows" in str(checks.get("warnings", [])))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_split_reproducible_cic(cic_ids2017_style_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="capture_file",
        label_column="label",
    )
    r1 = Splitter(cfg).split(cic_ids2017_style_df.copy())
    r2 = Splitter(cfg).split(cic_ids2017_style_df.copy())

    pd.testing.assert_frame_equal(r1.train_df, r2.train_df)
    pd.testing.assert_frame_equal(r1.val_df, r2.val_df)
    pd.testing.assert_frame_equal(r1.test_df, r2.test_df)
    assert r1.manifest == r2.manifest


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def test_manifest_written_to_json(cic_ids2017_style_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group", group_column="capture_file",
        label_column="label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(cic_ids2017_style_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/split_manifest.json"
        with open(path, "w") as f:
            json.dump(result.manifest.to_dict(), f, indent=2)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["strategy"] == "group"
        assert loaded["n_train"] + loaded["n_val"] + loaded["n_test"] == loaded["total_rows"]
        assert loaded["leakage_checks"]["passed"] is True


# ---------------------------------------------------------------------------
# Validation / test separation
# ---------------------------------------------------------------------------

def test_validation_and_test_are_separate(cic_ids2017_style_df: pd.DataFrame) -> None:
    cfg = SplitConfig(
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_seed=42, strategy="group", group_column="capture_file",
        label_column="label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(cic_ids2017_style_df)

    assert set(result.manifest.val_indices).isdisjoint(
        set(result.manifest.test_indices)
    )
    assert len(result.val_df) > 0
    assert len(result.test_df) > 0


# ---------------------------------------------------------------------------
# Unavoidable limitations documentation
# ---------------------------------------------------------------------------

def test_limitations_documentation() -> None:
    """Documents the known limitations of the splitting approach.

    This test serves as a living record of constraints that callers must
    handle outside the splitter:
    """
    limitations = [
        "Group splitting requires a stable group identifier column; without "
        "it, the splitter falls back to random/temporal only.",
        "Temporal splitting requires a reliable timestamp column; clock skew "
        "or missing timestamps degrade temporal isolation.",
        "Duplicate overlap detection uses row-content hashing; very large "
        "datasets may incur memory cost.",
        "Small groups (< min_group_size) may be merged into train, slightly "
        "altering ratios.",
        "Class imbalance in tiny groups can produce empty val/test splits "
        "for rare classes; stratified random splitting mitigates but does not "
        "eliminate this.",
    ]

    # All limitations are documented as non-empty strings.
    assert all(isinstance(l, str) and len(l) > 0 for l in limitations)
    assert len(limitations) == 5
