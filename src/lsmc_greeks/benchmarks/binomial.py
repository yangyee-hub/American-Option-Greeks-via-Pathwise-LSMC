"""Binomial-tree benchmark for 1D American put validation."""

from __future__ import annotations

import numpy as np


def american_put_binomial(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_steps: int = 1000,
) -> float:
    """Price a 1D American put using a Cox-Ross-Rubinstein tree."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    dt = maturity / n_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up
    discount = np.exp(-rate * dt)
    p = (np.exp(rate * dt) - down) / (up - down)

    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability is outside (0, 1); increase n_steps.")

    terminal_j = np.arange(n_steps + 1)
    terminal_spots = spot * (up ** terminal_j) * (down ** (n_steps - terminal_j))
    values = np.maximum(strike - terminal_spots, 0.0)

    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        spots = spot * (up ** j) * (down ** (step - j))
        continuation = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        exercise = np.maximum(strike - spots, 0.0)
        values = np.maximum(exercise, continuation)

    return float(values[0])


def american_put_delta_binomial(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_steps: int = 1000,
    bump: float = 0.25,
) -> float:
    """Central-difference delta built on the binomial benchmark price."""
    up_price = american_put_binomial(spot + bump, strike, rate, sigma, maturity, n_steps=n_steps)
    down_price = american_put_binomial(spot - bump, strike, rate, sigma, maturity, n_steps=n_steps)
    return float((up_price - down_price) / (2.0 * bump))
