"""Reproducibility utilities for Python, NumPy and PyTorch.

Provides seed-setting and deterministic configuration helpers so that Kaggle
training runs are reproducible where practical.
"""

from __future__ import annotations

import os
import random
from typing import Optional

try:  # NumPy is optional in the test environment
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:  # PyTorch is optional in the test environment
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set seeds for Python, NumPy and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: When True, enable PyTorch deterministic algorithms and
            set CuDNN deterministic/benchmark flags where practical.

    Returns:
        The seed that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _HAS_NUMPY:
        np.random.seed(seed)  # type: ignore[attr-defined]

    if _HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore[attr-defined]
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]

    if deterministic and _HAS_TORCH:
        # Use try/except because deterministic algorithm flags differ by version.
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass

    return seed


def is_deterministic() -> bool:
    """Return whether PyTorch is currently configured deterministically."""
    if not _HAS_TORCH:
        return False
    try:
        return bool(torch.are_deterministic_algorithms_enabled())  # type: ignore[attr-defined]
    except Exception:
        return False


def get_seeds() -> dict:
    """Return a snapshot of the currently configured seeds (best-effort)."""
    snapshot: dict = {"python_hash_seed": os.environ.get("PYTHONHASHSEED")}
    if _HAS_NUMPY:
        snapshot["numpy_seed"] = int(np.random.get_state()[1][0])  # type: ignore[attr-defined]
    if _HAS_TORCH:
        snapshot["torch_seed"] = int(torch.initial_seed())  # type: ignore[attr-defined]
    return snapshot


def make_deterministic_context(seed: int = 42) -> dict:
    """Convenience: set seed and return a reproducibility snapshot.

    Useful at the top of a Kaggle notebook cell to lock down randomness before
    any model construction.
    """
    set_seed(seed, deterministic=True)
    return {
        "seed": seed,
        "deterministic": True,
        "seeds": get_seeds(),
        "deterministic_mode": is_deterministic(),
    }
