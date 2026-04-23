"""Longstaff-Schwartz pricing routines for American puts."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .models import GBMParams, simulate_gbm_paths
from .payoffs import european_put_payoff, put_intrinsic
from .utils import normal_cdf, standard_error, to_serializable_dict


@dataclass(frozen=True)
class LSMCConfig:
    n_steps_per_year: int = 50
    n_paths: int = 100_000
    basis_degree: int = 2
    ridge_scale: float = 1e-6
    antithetic: bool = True
    seed: int | None = 42


@dataclass
class LSMCResult:
    american_price: float
    european_price: float
    std_error: float
    runtime_sec: float
    diagnostics: dict[str, object] = field(default_factory=dict)


def bs_european_put(spot: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    """Closed-form Black-Scholes price for a European put."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return float(strike * np.exp(-rate * maturity) * normal_cdf(-d2) - spot * normal_cdf(-d1))


def laguerre_basis(spot: np.ndarray, strike: float, degree: int = 2) -> np.ndarray:
    """Weighted Laguerre basis used in LS2001 for the American put example.

    The returned design matrix always includes a constant column, then the
    weighted Laguerre terms up to the requested degree:

    - degree=0: [1, L0]
    - degree=1: [1, L0, L1]
    - degree=2: [1, L0, L1, L2]
    - degree=3: [1, L0, L1, L2, L3]
    """
    if degree < 0 or degree > 3:
        raise ValueError("basis degree must be between 0 and 3 for this project baseline.")

    x = np.clip(spot / strike, 1e-4, 100.0)
    exp_term = np.exp(-x / 2.0)
    l0 = exp_term
    l1 = exp_term * (1.0 - x)
    l2 = exp_term * (1.0 - 2.0 * x + 0.5 * x**2)
    columns = [np.ones_like(x), l0]
    if degree >= 1:
        columns.append(l1)
    if degree >= 2:
        columns.append(l2)
    if degree >= 3:
        l3 = exp_term * (1.0 - 3.0 * x + 1.5 * x**2 - (x**3) / 6.0)
        columns.append(l3)
    return np.column_stack(columns)


def _fit_continuation_values(x: np.ndarray, y: np.ndarray, ridge_scale: float) -> np.ndarray:
    """Fit continuation values using least squares with a small ridge fallback."""
    try:
        coeffs = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x @ coeffs
        if not np.all(np.isfinite(fitted)):
            raise np.linalg.LinAlgError("non-finite continuation values")
        return fitted
    except np.linalg.LinAlgError:
        lam = ridge_scale * float((x.T @ x).diagonal().mean())
        xtx = x.T @ x + lam * np.eye(x.shape[1])
        coeffs = np.linalg.solve(xtx, x.T @ y)
        return x @ coeffs


def lsm_american_put(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    config: LSMCConfig | None = None,
    rng: np.random.Generator | None = None,
    normal_draws: np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> LSMCResult:
    """Price an American put with Longstaff-Schwartz Monte Carlo."""
    config = config or LSMCConfig()
    n_steps = max(int(round(maturity * config.n_steps_per_year)), 1)
    dt = maturity / n_steps
    discount = np.exp(-rate * dt)

    params = GBMParams(spot=spot, rate=rate, sigma=sigma)
    if rng is None and config.seed is not None and normal_draws is None:
        rng = np.random.default_rng(config.seed)

    start = perf_counter()
    paths = simulate_gbm_paths(
        params=params,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=config.n_paths,
        rng=rng,
        normal_draws=normal_draws,
        antithetic=config.antithetic,
    )

    cashflows = put_intrinsic(paths[:, -1], strike).astype(float)
    exercise_index = np.full(config.n_paths, n_steps, dtype=int)

    for t in range(n_steps - 1, 0, -1):
        cashflows *= discount
        spot_t = paths[:, t]
        intrinsic = put_intrinsic(spot_t, strike)
        itm = intrinsic > 0.0
        if not np.any(itm):
            continue

        design = laguerre_basis(spot_t[itm], strike=strike, degree=config.basis_degree)
        continuation = _fit_continuation_values(design, cashflows[itm], config.ridge_scale)
        continuation = np.where(np.isfinite(continuation), continuation, 0.0)

        exercise_now = intrinsic[itm] >= continuation
        itm_indices = np.flatnonzero(itm)
        chosen = itm_indices[exercise_now]
        cashflows[itm] = np.where(exercise_now, intrinsic[itm], cashflows[itm])
        exercise_index[chosen] = t

    discounted_cashflows = cashflows * discount
    runtime = perf_counter() - start

    result = LSMCResult(
        american_price=float(discounted_cashflows.mean()),
        european_price=float(np.exp(-rate * maturity) * european_put_payoff(paths[:, -1], strike).mean()),
        std_error=standard_error(discounted_cashflows),
        runtime_sec=runtime,
    )
    if return_diagnostics:
        result.diagnostics = {
            "config": to_serializable_dict(config),
            "paths": paths,
            "exercise_index": exercise_index,
            "discounted_cashflows": discounted_cashflows,
        }
    return result
