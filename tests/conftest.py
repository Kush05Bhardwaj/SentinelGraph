"""Pytest configuration and shared fixtures for SentinelGraph tests."""

import pytest


@pytest.fixture
def sample_network_flow() -> dict:
    """Return a minimal sample network flow record."""
    return {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 12345,
        "dst_port": 443,
        "protocol": "tcp",
        "bytes_sent": 1024,
        "bytes_received": 2048,
        "duration": 1.5,
    }


@pytest.fixture
def health_response() -> dict:
    """Return a minimal health check response."""
    return {"status": "ok"}
