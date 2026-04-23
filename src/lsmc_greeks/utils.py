"""Utility helpers used across pricing, benchmarking, and estimator modules."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from scipy.stats import norm


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF via SciPy."""
    x_array = np.asarray(x, dtype=float)
    values = norm.cdf(x_array)
    if np.isscalar(x):
        return float(values)
    return values


def to_serializable_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclasses and simple objects to plain dictionaries."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Unsupported object type: {type(obj)!r}")


def standard_error(samples: np.ndarray) -> float:
    """Monte Carlo standard error using sample standard deviation."""
    if samples.size <= 1:
        return 0.0
    return float(samples.std(ddof=1) / np.sqrt(samples.size))
