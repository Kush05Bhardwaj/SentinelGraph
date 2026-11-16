"""Integration tests for the feature engineering pipeline.

Covers:
    * End-to-end train -> test pipeline
    * Save/load feature configuration roundtrip
    * Config-driven behavior (custom zero-duration fallback)
    * No invalid numerical values across train and test
    * Feature schema persisted and reloaded
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.feature_engineering import (
    FeatureEngineer,
    FeatureEngineeringConfig,
    DurationConfig,
    load_config,
)


@pytest.fixture
def train_df() -> pd.DataFrame:
    np.random.seed(0)
    n = 100
    return pd.DataFrame(
        {
            "Flow ID": [f"f{i}" for i in range(n)],
            "Src IP": [f"10.0.0.{i % 50}" for i in range(n)],
            "Dst IP": [f"10.0.0.{(i + 20) % 50}" for i in range(n)],
            "Src Port": [1000 + (i % 100) for i in range(n)],
            "Dst Port": [[443, 80, 8080, 22][i % 4] for i in range(n)],
            "Protocol": np.random.choice(["tcp", "udp", "icmp"], size=n),
            "Flow Duration": np.random.exponential(2, size=n).round(3),
            "Total Fwd Packets": np.random.randint(1, 100, size=n),
            "Total Backward Packets": np.random.randint(1, 100, size=n),
            "Total Length of Fwd Packets": np.random.randint(100, 5000, size=n),
            "Total Length of Bwd Packets": np.random.randint(100, 5000, size=n),
            "FIN Flag Count": np.random.randint(0, 2, size=n),
            "SYN Flag Count": np.random.randint(0, 2, size=n),
            "RST Flag Count": np.random.randint(0, 2, size=n),
            "PSH Flag Count": np.random.randint(0, 2, size=n),
            "ACK Flag Count": np.random.randint(0, 2, size=n),
            "URG Flag Count": np.random.randint(0, 2, size=n),
            "Label": np.random.choice(["BENIGN", "ATTACK"], size=n),
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    np.random.seed(1)
    n = 30
    return pd.DataFrame(
        {
            "Flow ID": [f"t{i}" for i in range(n)],
            "Src IP": [f"10.0.0.{i % 50}" for i in range(n)],
            "Dst IP": [f"10.0.0.{(i + 20) % 50}" for i in range(n)],
            "Src Port": [2000 + (i % 100) for i in range(n)],
            "Dst Port": [[443, 80, 8080, 22][i % 4] for i in range(n)],
            "Protocol": np.random.choice(["tcp", "udp", "icmp"], size=n),
            "Flow Duration": np.random.exponential(2, size=n).round(3),
            "Total Fwd Packets": np.random.randint(1, 100, size=n),
            "Total Backward Packets": np.random.randint(1, 100, size=n),
            "Total Length of Fwd Packets": np.random.randint(100, 5000, size=n),
            "Total Length of Bwd Packets": np.random.randint(100, 5000, size=n),
            "FIN Flag Count": np.random.randint(0, 2, size=n),
            "SYN Flag Count": np.random.randint(0, 2, size=n),
            "RST Flag Count": np.random.randint(0, 2, size=n),
            "PSH Flag Count": np.random.randint(0, 2, size=n),
            "ACK Flag Count": np.random.randint(0, 2, size=n),
            "URG Flag Count": np.random.randint(0, 2, size=n),
            "Label": np.random.choice(["BENIGN", "ATTACK"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def test_train_test_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    train_result = engineer.fit_transform(train_df.copy())
    test_result = engineer.transform(test_df.copy())

    # Feature matrices are aligned (same columns, same order).
    assert list(train_result.X.columns) == list(test_result.X.columns)

    # No invalid values in either.
    for res in [train_result, test_result]:
        numeric = res.X.select_dtypes(include=[np.number])
        assert not np.isinf(numeric).any().any()
        assert not numeric.isna().any().any()

    # Raw columns separated from engineered.
    assert "Label" in train_result.raw.columns
    assert "Label" not in train_result.X.columns

    # Schema is shared.
    assert train_result.schema is test_result.schema


# ---------------------------------------------------------------------------
# Save / load config roundtrip
# ---------------------------------------------------------------------------

def test_save_load_config_roundtrip(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    train_result = engineer.fit_transform(train_df.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = os.path.join(tmpdir, "feature_engineering")
        engineer.save_config(artifact_dir)

        assert os.path.isfile(os.path.join(artifact_dir, "feature_schema.json"))
        assert os.path.isfile(os.path.join(artifact_dir, "feature_lookups.json"))

        loaded = load_config(artifact_dir)
        schema = loaded["schema"]
        lookups = loaded["lookups"]

        # Schema preserved.
        assert schema.version == "1.0"
        assert len(schema.feature_names()) == len(train_result.X.columns)
        assert list(schema.feature_names()) == list(train_result.X.columns)

        # Lookups preserved.
        assert "pair_freq" in lookups
        assert "src_freq" in lookups
        assert "protocol_mapping" in lookups


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

def test_custom_zero_duration_fallback(train_df: pd.DataFrame) -> None:
    cfg = FeatureEngineeringConfig(
        duration=DurationConfig(zero_duration_fallback=999.0)
    )
    engineer = FeatureEngineer(cfg)
    # Inject zero-duration rows explicitly (random data rarely has exact 0).
    train_df.loc[0, "Flow Duration"] = 0.0
    train_df.loc[1, "Flow Duration"] = 0.0
    result = engineer.fit_transform(train_df.copy())

    # Zero-duration rows use fallback 999.0.
    for idx in [0, 1]:
        assert result.X.loc[idx, "flow_duration"] == 999.0


# ---------------------------------------------------------------------------
# Consistency across train and test (no refit)
# ---------------------------------------------------------------------------

def test_protocol_mapping_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    engineer = FeatureEngineer()
    engineer.fit(train_df.copy())
    train_result = engineer.transform(train_df.copy())
    test_result = engineer.transform(test_df.copy())

    # Same protocol -> same code in both train and test.
    proto_map = {}
    for _, row in train_df.iterrows():
        proto_map[row["Protocol"]] = train_result.X.loc[row.name, "protocol_code"]
    for _, row in test_df.iterrows():
        if row["Protocol"] in proto_map:
            idx = test_df.index[test_df["Protocol"] == row["Protocol"]][0]
            assert test_result.X.loc[idx, "protocol_code"] == proto_map[row["Protocol"]]
