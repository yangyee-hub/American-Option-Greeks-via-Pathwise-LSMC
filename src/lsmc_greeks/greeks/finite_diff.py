"""Finite-difference estimators used as the baseline Greek method."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ..pricer import LSMCConfig, lsm_american_put


def estimate_delta_fd(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    config: LSMCConfig | None = None,
    bump: float = 0.25,
    seed: int | None = 42,
) -> dict[str, object]:
    """Central finite-difference delta estimate using common random numbers."""
    config = config or LSMCConfig()
    n_steps = max(int(round(maturity * config.n_steps_per_year)), 1)
    rng = np.random.default_rng(seed if seed is not None else config.seed)

    if config.antithetic:
        base_draws = rng.standard_normal((config.n_paths // 2, n_steps))
        shared_draws = np.concatenate([base_draws, -base_draws], axis=0)
    else:
        shared_draws = rng.standard_normal((config.n_paths, n_steps))

    start = perf_counter()
    up = lsm_american_put(
        spot=spot + bump,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        config=config,
        normal_draws=shared_draws,
    )
    down = lsm_american_put(
        spot=spot - bump,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        config=config,
        normal_draws=shared_draws,
    )
    runtime = perf_counter() - start

    return {
        "estimate": (up.american_price - down.american_price) / (2.0 * bump),
        "std_error": None,
        "runtime_sec": runtime,
        "metadata": {
            "bump": bump,
            "method": "central_finite_difference",
            "common_random_numbers": True,
            "up_price": up.american_price,
            "down_price": down.american_price,
        },
    }
