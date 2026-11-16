"""Configuration for the feature engineering pipeline.

All settings are explicit and serializable so feature generation is
deterministic and reproducible across training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DurationConfig:
    """Configuration for flow-duration derivation."""

    source_column: Optional[str] = "Flow Duration"
    timestamp_column: Optional[str] = "Timestamp"
    flow_id_column: Optional[str] = "Flow ID"
    zero_duration_fallback: float = 1e-6  # value used when duration is 0 or missing


@dataclass
class RateConfig:
    """Configuration for per-second rate features."""

    packet_rate_column: str = "packets_per_sec"
    byte_rate_column: str = "bytes_per_sec"
    guard_zero_duration: bool = True


@dataclass
class FlagConfig:
    """Configuration for TCP flag count features."""

    flag_columns: List[str] = field(
        default_factory=lambda: [
            "FIN Flag Count",
            "SYN Flag Count",
            "RST Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "URG Flag Count",
        ]
    )
    output_prefix: str = "flag_"


@dataclass
class PortConfig:
    """Configuration for destination-port features."""

    dst_port_column: Optional[str] = "Dst Port"
    well_known_max: int = 1023
    registered_max: int = 49151


@dataclass
class FrequencyConfig:
    """Configuration for connection-frequency features."""

    src_column: Optional[str] = "Src IP"
    dst_column: Optional[str] = "Dst Port"
    dst_ip_column: Optional[str] = "Dst IP"
    enabled: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Top-level feature engineering configuration."""

    duration: DurationConfig = field(default_factory=DurationConfig)
    rate: RateConfig = field(default_factory=RateConfig)
    flags: FlagConfig = field(default_factory=FlagConfig)
    port: PortConfig = field(default_factory=PortConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    include_raw_identifiers: bool = False  # keep src/dst IPs/ports as features?
    random_seed: int = 42
