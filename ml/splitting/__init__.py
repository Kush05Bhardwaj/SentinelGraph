"""Public API for the splitting package."""

from __future__ import annotations

from .config import SplitConfig
from .exceptions import SplitError, SplitConfigError, SplitLeakageError
from .manifest import SplitManifest
from .splitter import Splitter, SplitResult

__all__ = [
    "Splitter",
    "SplitResult",
    "SplitConfig",
    "SplitManifest",
    "SplitError",
    "SplitConfigError",
    "SplitLeakageError",
]
