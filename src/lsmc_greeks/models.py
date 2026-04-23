"""Model definitions and GBM path simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBMParams:
    spot: float
    rate: float
    sigma: float


def simulate_gbm_paths(
    params: GBMParams,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator | None = None,
    normal_draws: np.ndarray | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate GBM price paths under the risk-neutral measure.

    Parameters
    ----------
    params:
        GBM model parameters.
    maturity:
        Time to expiry in years.
    n_steps:
        Number of exercise dates / time steps.
    n_paths:
        Number of simulated paths.
    rng:
        Optional NumPy generator used only when `normal_draws` is not supplied.
    normal_draws:
        Optional shock matrix of shape `(n_paths, n_steps)` or
        `(n_paths // 2, n_steps)` when `antithetic=True`.
    antithetic:
        When True and `normal_draws` is omitted, generate antithetic shocks.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 1:
        raise ValueError("n_paths must be greater than 1.")

    dt = maturity / n_steps

    if normal_draws is None:
        if rng is None:
            rng = np.random.default_rng(42)
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True.")
            half = n_paths // 2
            base_draws = rng.standard_normal((half, n_steps))
            shocks = np.concatenate([base_draws, -base_draws], axis=0)
        else:
            shocks = rng.standard_normal((n_paths, n_steps))
    else:
        shocks = np.asarray(normal_draws, dtype=float)
        if antithetic and shocks.shape == (n_paths // 2, n_steps):
            shocks = np.concatenate([shocks, -shocks], axis=0)
        if shocks.shape != (n_paths, n_steps):
            raise ValueError(
                "normal_draws must have shape (n_paths, n_steps) or "
                "(n_paths // 2, n_steps) when antithetic=True."
            )

    drift = (params.rate - 0.5 * params.sigma**2) * dt
    diffusion = params.sigma * np.sqrt(dt) * shocks
    increments = drift + diffusion

    log_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(params.spot)
    log_paths[:, 1:] = np.log(params.spot) + np.cumsum(increments, axis=1)
    return np.exp(log_paths)
