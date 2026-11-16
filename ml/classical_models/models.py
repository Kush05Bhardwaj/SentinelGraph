"""Model initialization for classical ML classifiers.

Provides a registry of sklearn-compatible classifiers and a factory function
to instantiate them from a :class:`ModelConfig`. Supports both binary and
multiclass classification tasks.

The registry intentionally avoids hardcoding any dataset-specific paths or
column names; all configuration is driven by the provided :class:`ModelConfig`.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Type

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .exceptions import ModelConfigurationError

try:  # XGBoost is optional
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on environment
    XGBClassifier = None  # type: ignore[assignment, misc]
    _HAS_XGBOOST = False

try:  # LightGBM is optional
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LGBM = True
except Exception:  # pragma: no cover - depends on environment
    LGBMClassifier = None  # type: ignore[assignment, misc]
    _HAS_LGBM = False


# Registry of supported algorithms. Keys are lowercase names; values are the
# sklearn-compatible estimator classes.
MODEL_REGISTRY: Dict[str, Type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier,
    "svc": SVC,
    "knn": KNeighborsClassifier,
}

if _HAS_XGBOOST:
    MODEL_REGISTRY["xgboost"] = XGBClassifier  # type: ignore[assignment]
if _HAS_LGBM:
    MODEL_REGISTRY["lightgbm"] = LGBMClassifier  # type: ignore[assignment]


def get_model_class(algorithm: str) -> Type:
    """Return the estimator class for a given algorithm name.

    Args:
        algorithm: Algorithm name (case-insensitive), e.g. ``"random_forest"``.

    Returns:
        The estimator class.

    Raises:
        ModelConfigurationError: If the algorithm is not registered.
    """
    key = algorithm.lower().strip()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ModelConfigurationError(
            f"Unknown algorithm '{algorithm}'. Available: {available}"
        )
    return MODEL_REGISTRY[key]


def initialize_model(config: Any) -> Any:
    """Initialize a model from a :class:`ModelConfig`.

    The model is constructed with the hyperparameters from the config and a
    deterministic random seed where supported.

    Args:
        config: A :class:`ModelConfig` instance.

    Returns:
        An unfitted sklearn-compatible estimator instance.
    """
    model_class = get_model_class(config.algorithm)
    params = dict(config.hyperparameters)

    # Inspect the constructor signature to determine which seed/n_jobs
    # parameters the estimator accepts. sklearn estimators define these as
    # __init__ keyword arguments, not class attributes, so hasattr() does not
    # work here.
    try:
        valid_params = set(inspect.signature(model_class.__init__).parameters.keys())
    except (ValueError, TypeError):  # pragma: no cover - built-in C types
        valid_params = set()

    # Inject a deterministic random seed where the estimator supports it.
    for seed_param in ("random_state", "seed"):
        if seed_param in valid_params and seed_param not in params:
            params[seed_param] = config.random_seed
            break

    # Force single-threaded execution for deterministic results where supported.
    if "n_jobs" in valid_params and "n_jobs" not in params:
        params["n_jobs"] = 1

    return model_class(**params)


def list_available_algorithms() -> list:
    """Return a sorted list of available algorithm names."""
    return sorted(MODEL_REGISTRY.keys())
