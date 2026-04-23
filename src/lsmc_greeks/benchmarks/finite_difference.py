"""Finite-difference benchmark for 1D American put validation.

This module adapts the useful idea from the teammate notebook into the shared
`src/` package and uses SciPy's banded solver, which is the conventional
scientific-Python choice for this kind of tridiagonal linear system.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def american_put_finite_difference(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_space_steps: int = 400,
    n_time_steps_per_year: int = 4000,
    spot_max_multiplier: float = 4.0,
) -> float:
    """Price a 1D American put with a fully implicit finite-difference scheme."""
    if n_space_steps <= 1:
        raise ValueError("n_space_steps must be greater than 1.")
    if n_time_steps_per_year <= 0:
        raise ValueError("n_time_steps_per_year must be positive.")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive.")

    n_time_steps = max(int(round(n_time_steps_per_year * maturity)), 1)
    spot_max = spot_max_multiplier * strike
    d_spot = spot_max / n_space_steps
    dt = maturity / n_time_steps

    spot_grid = np.linspace(0.0, spot_max, n_space_steps + 1)
    values = np.maximum(strike - spot_grid, 0.0)
    intrinsic = values.copy()

    j = np.arange(1, n_space_steps, dtype=float)
    alpha = 0.5 * dt * sigma**2 * j**2
    beta = 0.5 * dt * rate * j

    lower = -(alpha - beta)[1:]
    diag = 1.0 + 2.0 * alpha + rate * dt
    upper = -(alpha + beta)[:-1]
    banded = np.zeros((3, n_space_steps - 1), dtype=float)
    banded[0, 1:] = upper
    banded[1, :] = diag
    banded[2, :-1] = lower

    values[0] = strike
    values[-1] = 0.0

    for _ in range(n_time_steps):
        rhs = values[1:-1].copy()
        rhs[0] -= (-(alpha[0] - beta[0])) * values[0]
        rhs[-1] -= (-(alpha[-1] + beta[-1])) * values[-1]

        continuation = solve_banded((1, 1), banded, rhs)
        values[1:-1] = np.maximum(continuation, intrinsic[1:-1])

    idx = spot / d_spot
    lo = int(np.floor(idx))
    lo = int(np.clip(lo, 0, n_space_steps))
    hi = int(np.clip(lo + 1, 0, n_space_steps))
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def american_put_delta_finite_difference(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    n_space_steps: int = 400,
    n_time_steps_per_year: int = 4000,
    bump: float = 0.25,
) -> float:
    """Central-difference delta built on the finite-difference benchmark price."""
    up_price = american_put_finite_difference(
        spot=spot + bump,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_space_steps=n_space_steps,
        n_time_steps_per_year=n_time_steps_per_year,
    )
    down_price = american_put_finite_difference(
        spot=spot - bump,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        n_space_steps=n_space_steps,
        n_time_steps_per_year=n_time_steps_per_year,
    )
    return float((up_price - down_price) / (2.0 * bump))
