"""Payoff helpers used by pricing and validation code."""

from __future__ import annotations

import numpy as np


def put_intrinsic(spot: np.ndarray | float, strike: float) -> np.ndarray | float:
    """Intrinsic value of a vanilla put option."""
    return np.maximum(strike - spot, 0.0)


def european_put_payoff(terminal_spot: np.ndarray, strike: float) -> np.ndarray:
    """Terminal payoff of a vanilla European put."""
    return np.maximum(strike - terminal_spot, 0.0)
