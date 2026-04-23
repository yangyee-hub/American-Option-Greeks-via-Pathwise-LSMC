"""Mixed Greek estimators combining Pathwise and Likelihood Ratio methods."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from ..pricer import LSMCConfig, lsm_american_put
from ..utils import standard_error

def estimate_greeks_mixed(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    config: LSMCConfig | None = None,
    seed: int | None = 42,
    normal_draws: np.ndarray | None = None,
) -> dict[str, object]:
    """Estimate Greeks using a Mixed Estimator strategy.
    
    Delta is estimated using the Pathwise method for minimal variance.
    Gamma is estimated using the Likelihood Ratio (LR) method to correctly
    handle the discontinuities where the pure pathwise method breaks down.
    """
    config = config or LSMCConfig()
    effective_seed = seed if seed is not None else config.seed

    start = perf_counter()
    pricing_result = lsm_american_put(
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        config=config,
        rng=None if normal_draws is not None else np.random.default_rng(effective_seed),
        normal_draws=normal_draws,
        return_diagnostics=True,
    )

    diagnostics = pricing_result.diagnostics
    paths = np.asarray(diagnostics["paths"], dtype=float)
    exercise_index = np.asarray(diagnostics["exercise_index"], dtype=int)
    
    n_paths = paths.shape[0]
    n_steps = max(int(round(maturity * config.n_steps_per_year)), 1)
    dt = maturity / n_steps

    exercised_spot = paths[np.arange(n_paths), exercise_index]
    intrinsic = np.maximum(strike - exercised_spot, 0.0)
    discounts = np.exp(-rate * dt * exercise_index)
    discounted_payoff = discounts * intrinsic

    # ---------------------------------------------------------
    # 1. Delta: Pathwise Estimator (Matched with JH's logic)
    # ---------------------------------------------------------
    pathwise_delta_samples = discounts * np.where(intrinsic > 0.0, -(exercised_spot / spot), 0.0)

    # ---------------------------------------------------------
    # 2. Gamma: Likelihood Ratio Estimator (MH's logic)
    # ---------------------------------------------------------
    S1 = paths[:, 1]
    Z1 = (np.log(S1 / spot) - (rate - 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))
    score_gamma = (Z1**2 - 1.0 - Z1 * sigma * np.sqrt(dt)) / (spot**2 * sigma**2 * dt)
    
    lr_gamma_samples = discounted_payoff * score_gamma

    runtime = perf_counter() - start

    return {
        "delta_estimate": float(pathwise_delta_samples.mean()),
        "delta_std_error": standard_error(pathwise_delta_samples),
        "gamma_estimate": float(lr_gamma_samples.mean()),
        "gamma_std_error": standard_error(lr_gamma_samples),
        "runtime_sec": runtime,
        "metadata": {
            "method": "mixed_pathwise_lr",
            "seed": effective_seed,
            "n_paths": n_paths,
        }
    }