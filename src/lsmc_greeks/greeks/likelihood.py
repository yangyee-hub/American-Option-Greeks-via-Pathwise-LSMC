"""Likelihood Ratio Greek estimators for the LSMC project."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from ..pricer import LSMCConfig, lsm_american_put
from ..utils import standard_error

def estimate_greeks_lr(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    config: LSMCConfig | None = None,
    seed: int | None = 42,
    normal_draws: np.ndarray | None = None,
) -> dict[str, object]:
    """Estimate Delta and Gamma using the Likelihood Ratio (LR) method.
    
    This method differentiates the probability density function rather than 
    the payoff function. It handles discontinuities in the payoff and the 
    early exercise boundary well, but generally exhibits higher variance 
    than pathwise methods.
    """
    config = config or LSMCConfig()
    effective_seed = seed if seed is not None else config.seed

    start = perf_counter()
    # Call YY's pricer and extract the simulation diagnostics
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

    # ---------------------------------------------------------
    # 1. Recover the first step standard normal draw (Z1)
    # ---------------------------------------------------------
    S1 = paths[:, 1]
    Z1 = (np.log(S1 / spot) - (rate - 0.5 * sigma**2) * dt) / (sigma * np.sqrt(dt))

    # ---------------------------------------------------------
    # 2. Compute Score Functions (Broadie-Glasserman 1996)
    # ---------------------------------------------------------
    score_delta = Z1 / (spot * sigma * np.sqrt(dt))
    score_gamma = (Z1**2 - 1.0 - Z1 * sigma * np.sqrt(dt)) / (spot**2 * sigma**2 * dt)

    # ---------------------------------------------------------
    # 3. Apply Scores to Discounted Payoffs
    # ---------------------------------------------------------
    exercised_spot = paths[np.arange(n_paths), exercise_index]
    intrinsic = np.maximum(strike - exercised_spot, 0.0)
    discounts = np.exp(-rate * dt * exercise_index)
    discounted_payoff = discounts * intrinsic

    delta_samples = discounted_payoff * score_delta
    gamma_samples = discounted_payoff * score_gamma

    runtime = perf_counter() - start

    return {
        "delta_estimate": float(delta_samples.mean()),
        "delta_std_error": standard_error(delta_samples),
        "gamma_estimate": float(gamma_samples.mean()),
        "gamma_std_error": standard_error(gamma_samples),
        "runtime_sec": runtime,
        "metadata": {
            "method": "likelihood_ratio",
            "seed": effective_seed,
            "n_paths": n_paths,
        }
    }