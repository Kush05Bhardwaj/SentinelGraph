"""Unit tests for the SentinelGraph feature engineering pipeline.

Covers:
    * Flow duration derivation + zero-duration handling
    * Forward/backward packet & byte counts
    * Packets/bytes per second (division-by-zero guard)
    * TCP flag counts
    * Protocol representation
    * Destination port features (raw + category)
    * Connection frequency (fit-on-train leakage safety)
    * Source/destination communication statistics
    * Stable feature schema & ordering
    * Deterministic behavior
    * Not-fitted error
    * Missing columns handled gracefully
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    FeatureEngineer,
    FeatureEngineeringConfig,
    DurationConfig,
    RateConfig,
    FlagConfig,
    PortConfig,
    FrequencyConfig,
)
from ml.feature_engineering.exceptions import FeatureNotFittedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["f1", "f2", "f3", "f4"],
            "Src IP": ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.3"],
            "Dst IP": ["10.0.0.2", "10.0.0.3", "10.0.0.2", "10.0.0.4"],
            "Src Port": [1000, 2000, 1000, 3000],
            "Dst Port": [443, 80, 443, 8080],
            "Protocol": ["tcp", "udp", "tcp", "icmp"],
            "Flow Duration": [1.0, 2.0, 0.0, 4.0],  # f3 is zero-duration
            "Total Fwd Packets": [10, 20, 5, 30],
            "Total Backward Packets": [5, 10, 2, 15],
            "Total Length of Fwd Packets": [1000, 2000, 500, 3000],
            "Total Length of Bwd Packets": [500, 1000, 200, 1500],
            "FIN Flag Count": [1, 0, 1, 0],
            "SYN Flag Count": [1, 1, 0, 0],
            "RST Flag Count": [0, 0, 0, 1],
            "PSH Flag Count": [0, 0, 0, 0],
            "ACK Flag Count": [1, 0, 1, 0],
            "URG Flag Count": [0, 0, 0, 0],
            "Label": ["BENIGN", "ATTACK", "BENIGN", "ATTACK"],
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow ID": ["f5", "f6"],
            "Src IP": ["10.0.0.5", "10.0.0.1"],  # 10.0.0.1 seen in train
            "Dst IP": ["10.0.0.6", "10.0.0.2"],  # 10.0.0.2 seen in train
            "Src Port": [4000, 1000],
            "Dst Port": [22, 443],
            "Protocol": ["tcp", "tcp"],  # tcp seen in train
            "Flow Duration": [3.0, 0.0],
            "Total Fwd Packets": [7, 8],
            "Total Backward Packets": [3, 4],
            "Total Length of Fwd Packets": [700, 800],
            "Total Length of Bwd Packets": [300, 400],
            "FIN Flag Count": [0, 1],
            "SYN Flag Count": [1, 1],
            "RST Flag Count": [0, 0],
            "PSH Flag Count": [0, 0],
            "ACK Flag Count": [1, 0],
            "URG Flag Count": [0, 0],
            "Label": ["ATTACK", "BENIGN"],
        }
    )


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """A df with only a few columns — ensures graceful degradation."""
    return pd.DataFrame(
        {
            "Flow ID": ["m1"],
            "Src IP": ["10.0.0.9"],
            "Dst IP": ["10.0.0.10"],
            "Label": ["BENIGN"],
        }
    )


# ---------------------------------------------------------------------------
# Basic feature generation
# ---------------------------------------------------------------------------

def test_engineered_features_present(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    expected = {
        "flow_duration",
        "fwd_packets",
        "bwd_packets",
        "fwd_bytes",
        "bwd_bytes",
        "total_packets",
        "total_bytes",
        "packets_per_sec",
        "bytes_per_sec",
        "flag_fin",
        "flag_syn",
        "flag_rst",
        "flag_psh",
        "flag_ack",
        "flag_urg",
        "protocol_code",
        "dst_port",
        "dst_port_category",
        "src_conn_freq",
        "dst_conn_freq",
        "pair_conn_freq",
        "src_avg_bytes",
        "dst_avg_bytes",
    }
    assert expected.issubset(set(result.X.columns)), (
        f"Missing: {expected - set(result.X.columns)}"
    )


def test_label_not_in_features(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())
    assert "Label" not in result.X.columns


def test_identifiers_excluded_by_default(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())
    for col in ["Src IP", "Dst IP", "Dst Port", "Protocol"]:
        assert col not in result.X.columns


def test_raw_columns_contain_label(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())
    assert "Label" in result.raw.columns


# ---------------------------------------------------------------------------
# Flow duration & zero-duration handling
# ---------------------------------------------------------------------------

def test_zero_duration_guarded(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    # f3 has duration 0 -> rate features must not be inf/nan.
    assert not np.isinf(result.X["packets_per_sec"]).any()
    assert not np.isnan(result.X["packets_per_sec"]).any()
    assert not np.isinf(result.X["bytes_per_sec"]).any()
    assert not np.isnan(result.X["bytes_per_sec"]).any()

    # flow_duration for f3 should be the fallback (non-zero).
    assert result.X.loc[2, "flow_duration"] > 0


def test_no_invalid_numerical_values(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())
    numeric = result.X.select_dtypes(include=[np.number])
    assert not np.isinf(numeric).any().any()
    assert not numeric.isna().any().any()


# ---------------------------------------------------------------------------
# Packet / byte counts
# ---------------------------------------------------------------------------

def test_packet_byte_counts(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    pd.testing.assert_series_equal(
        result.X["fwd_packets"],
        pd.Series([10.0, 20.0, 5.0, 30.0], name="fwd_packets"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        result.X["bwd_packets"],
        pd.Series([5.0, 10.0, 2.0, 15.0], name="bwd_packets"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        result.X["total_packets"],
        pd.Series([15.0, 30.0, 7.0, 45.0], name="total_packets"),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        result.X["total_bytes"],
        pd.Series([1500.0, 3000.0, 700.0, 4500.0], name="total_bytes"),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Rate features
# ---------------------------------------------------------------------------

def test_packets_per_sec(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    # f1: 15 packets / 1.0 sec = 15
    assert result.X.loc[0, "packets_per_sec"] == pytest.approx(15.0)
    # f3: 7 packets / fallback (1e-6) -> large but finite
    assert math.isfinite(result.X.loc[2, "packets_per_sec"])


# ---------------------------------------------------------------------------
# TCP flags
# ---------------------------------------------------------------------------

def test_tcp_flag_counts(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    assert result.X.loc[0, "flag_fin"] == 1.0
    assert result.X.loc[0, "flag_syn"] == 1.0
    assert result.X.loc[0, "flag_rst"] == 0.0


# ---------------------------------------------------------------------------
# Protocol representation
# ---------------------------------------------------------------------------

def test_protocol_encoding(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    # protocol_code should be numeric
    assert pd.api.types.is_numeric_dtype(result.X["protocol_code"])
    # tcp appears in rows 0 and 2 -> same code
    assert result.X.loc[0, "protocol_code"] == result.X.loc[2, "protocol_code"]


def test_protocol_unknown_at_transform() -> None:
    train = pd.DataFrame(
        {
            "Src IP": ["10.0.0.1"],
            "Dst IP": ["10.0.0.2"],
            "Dst Port": [443],
            "Protocol": ["tcp"],
            "Flow Duration": [1.0],
            "Total Fwd Packets": [10],
            "Total Backward Packets": [5],
            "Total Length of Fwd Packets": [1000],
            "Total Length of Bwd Packets": [500],
            "FIN Flag Count": [0],
            "SYN Flag Count": [0],
            "RST Flag Count": [0],
            "PSH Flag Count": [0],
            "ACK Flag Count": [0],
            "URG Flag Count": [0],
            "Label": ["BENIGN"],
        }
    )
    test = pd.DataFrame(
        {
            "Src IP": ["10.0.0.3"],
            "Dst IP": ["10.0.0.4"],
            "Dst Port": [80],
            "Protocol": ["ipv6"],  # unseen at fit
            "Flow Duration": [1.0],
            "Total Fwd Packets": [10],
            "Total Backward Packets": [5],
            "Total Length of Fwd Packets": [1000],
            "Total Length of Bwd Packets": [500],
            "FIN Flag Count": [0],
            "SYN Flag Count": [0],
            "RST Flag Count": [0],
            "PSH Flag Count": [0],
            "ACK Flag Count": [0],
            "URG Flag Count": [0],
            "Label": ["ATTACK"],
        }
    )
    engineer = FeatureEngineer()
    engineer.fit(train)
    result = engineer.transform(test)
    # unseen protocol -> -1
    assert result.X.loc[0, "protocol_code"] == -1.0


# ---------------------------------------------------------------------------
# Destination port features
# ---------------------------------------------------------------------------

def test_dst_port_category(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    # f1 dst port 443 -> well-known (0)
    assert result.X.loc[0, "dst_port_category"] == 0.0
    # f4 dst port 8080 -> registered (1)
    assert result.X.loc[3, "dst_port_category"] == 1.0


# ---------------------------------------------------------------------------
# Connection frequency (leakage safety)
# ---------------------------------------------------------------------------

def test_connection_frequency_fit_on_train_only(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    engineer = FeatureEngineer()
    train_result = engineer.fit_transform(train_df.copy())
    test_result = engineer.transform(test_df.copy())

    # src_conn_freq learned from train: 10.0.0.1 appears twice in train.
    assert train_result.X.loc[0, "src_conn_freq"] == 2.0
    assert train_result.X.loc[2, "src_conn_freq"] == 2.0

    # At transform, test row f6 has src_ip 10.0.0.1 (seen in train) -> 2.
    assert test_result.X.loc[1, "src_conn_freq"] == 2.0
    # Test row f5 has src_ip 10.0.0.5 (unseen) -> 0.
    assert test_result.X.loc[0, "src_conn_freq"] == 0.0

    # pair_conn_freq: (10.0.0.1, 10.0.0.2) appears twice in train (f1 and f3).
    assert train_result.X.loc[0, "pair_conn_freq"] == 2.0


def test_no_target_leakage_from_frequency(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Frequency lookups must come from train, not from test labels."""
    engineer = FeatureEngineer()
    train_result = engineer.fit_transform(train_df.copy())
    test_result = engineer.transform(test_df.copy())

    # Train pair (10.0.0.1, 10.0.0.2) count is based solely on train rows.
    # f1 and f3 share this pair, so count is 2.
    assert train_result.X.loc[0, "pair_conn_freq"] == 2.0


# ---------------------------------------------------------------------------
# Source/destination communication statistics
# ---------------------------------------------------------------------------

def test_src_dst_avg_bytes(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(train_df.copy())

    # src_avg_bytes and dst_avg_bytes should be finite and >= 0.
    assert (result.X["src_avg_bytes"] >= 0).all()
    assert (result.X["dst_avg_bytes"] >= 0).all()
    assert np.isfinite(result.X["src_avg_bytes"]).all()
    assert np.isfinite(result.X["dst_avg_bytes"]).all()


# ---------------------------------------------------------------------------
# Missing columns / graceful degradation
# ---------------------------------------------------------------------------

def test_minimal_columns_no_crash(minimal_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    result = engineer.fit_transform(minimal_df.copy())

    # Should not raise; expected features present with defaults.
    assert "flow_duration" in result.X.columns
    assert "packets_per_sec" in result.X.columns
    assert "bytes_per_sec" in result.X.columns
    # No NaN/inf.
    numeric = result.X.select_dtypes(include=[np.number])
    assert not np.isinf(numeric).any().any()
    assert not numeric.isna().any().any()


# ---------------------------------------------------------------------------
# Stable schema & ordering
# ---------------------------------------------------------------------------

def test_stable_feature_schema_order(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    r1 = engineer.fit_transform(train_df.copy())
    r2 = engineer.fit_transform(train_df.copy())

    assert list(r1.X.columns) == list(r2.X.columns)
    assert list(r1.X.columns) == engineer._schema.feature_names()


def test_schema_documents_every_feature(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    engineer.fit_transform(train_df.copy())

    schema = engineer._schema
    assert len(schema.feature_entries) == len(schema.feature_names())
    for entry in schema.feature_entries:
        assert entry.name
        assert entry.derivation  # every feature documented
        assert entry.dtype


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic(train_df: pd.DataFrame) -> None:
    cfg = FeatureEngineeringConfig()
    r1 = FeatureEngineer(cfg).fit_transform(train_df.copy())
    r2 = FeatureEngineer(cfg).fit_transform(train_df.copy())
    pd.testing.assert_frame_equal(r1.X, r2.X)
    pd.testing.assert_frame_equal(r1.raw, r2.raw)


# ---------------------------------------------------------------------------
# Not fitted
# ---------------------------------------------------------------------------

def test_transform_before_fit_raises(train_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    with pytest.raises(FeatureNotFittedError):
        engineer.transform(train_df)


# ---------------------------------------------------------------------------
# Config variations
# ---------------------------------------------------------------------------

def test_include_raw_identifiers_opt_in(train_df: pd.DataFrame) -> None:
    cfg = FeatureEngineeringConfig(include_raw_identifiers=True)
    engineer = FeatureEngineer(cfg)
    result = engineer.fit_transform(train_df.copy())

    # With opt-in, identifiers appear in raw, not X.
    assert "Src IP" in result.raw.columns
    assert "Src IP" not in result.X.columns
