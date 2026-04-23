"""Pathwise Greek estimators for the LSMC project."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ..pricer import LSMCConfig, lsm_american_put
from ..utils import standard_error


def estimate_delta_pathwise(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    config: LSMCConfig | None = None,
    seed: int | None = 42,
    normal_draws: np.ndarray | None = None,
) -> dict[str, object]:
    """Estimate delta with a pathwise derivative under a fixed stopping rule.

    This estimator reuses the LSMC exercise policy from the pricing run and then
    differentiates the discounted intrinsic payoff path by path. The derivative
    of the stopping time is ignored, which is the standard first-order
    approximation used in the fixed-policy pathwise treatment.
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

    # Under GBM, dS_t / dS_0 = S_t / S_0. We apply that at the realized
    # stopping time while holding the LSMC exercise policy fixed.
    delta_samples = discounts * np.where(intrinsic > 0.0, -(exercised_spot / spot), 0.0)
    runtime = perf_counter() - start

    return {
        "estimate": float(delta_samples.mean()),
        "std_error": standard_error(delta_samples),
        "runtime_sec": runtime,
        "metadata": {
            "method": "pathwise_fixed_policy",
            "seed": effective_seed,
            "assumption": "LSMC stopping rule held fixed pathwise",
            "pricing_runtime_sec": pricing_result.runtime_sec,
            "exercise_index": exercise_index,
            "discounted_delta_samples": delta_samples,
        },
    }
