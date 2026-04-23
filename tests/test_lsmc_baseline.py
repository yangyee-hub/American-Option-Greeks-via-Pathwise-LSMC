"""Basic regression tests for the LSMC baseline package."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lsmc_greeks.benchmarks.binomial import american_put_binomial
from lsmc_greeks.benchmarks.finite_difference import (
    american_put_delta_finite_difference,
    american_put_finite_difference,
)
from lsmc_greeks.greeks.finite_diff import estimate_delta_fd
from lsmc_greeks.greeks.pathwise import estimate_delta_pathwise
from lsmc_greeks.models import GBMParams, simulate_gbm_paths
from lsmc_greeks.payoffs import put_intrinsic
from lsmc_greeks.pricer import LSMCConfig, bs_european_put, laguerre_basis, lsm_american_put


class TestLSMCBaseline(unittest.TestCase):
    def test_simulate_gbm_paths_shape(self) -> None:
        params = GBMParams(spot=40.0, rate=0.06, sigma=0.2)
        paths = simulate_gbm_paths(params, maturity=1.0, n_steps=5, n_paths=10)
        self.assertEqual(paths.shape, (10, 6))
        self.assertTrue(np.allclose(paths[:, 0], 40.0))

    def test_put_intrinsic(self) -> None:
        spots = np.array([36.0, 40.0, 44.0])
        intrinsic = put_intrinsic(spots, strike=40.0)
        np.testing.assert_allclose(intrinsic, np.array([4.0, 0.0, 0.0]))

    def test_laguerre_basis_degree_controls_column_count(self) -> None:
        spots = np.array([36.0, 40.0, 44.0])
        self.assertEqual(laguerre_basis(spots, strike=40.0, degree=0).shape, (3, 2))
        self.assertEqual(laguerre_basis(spots, strike=40.0, degree=1).shape, (3, 3))
        self.assertEqual(laguerre_basis(spots, strike=40.0, degree=2).shape, (3, 4))
        self.assertEqual(laguerre_basis(spots, strike=40.0, degree=3).shape, (3, 5))

    def test_lsm_price_near_reference_row(self) -> None:
        config = LSMCConfig(n_paths=20_000, seed=7)
        result = lsm_american_put(spot=36.0, strike=40.0, rate=0.06, sigma=0.2, maturity=1.0, config=config)
        self.assertAlmostEqual(result.american_price, 4.478, delta=0.12)
        self.assertGreater(result.american_price, result.european_price)

    def test_binomial_benchmark_is_reasonable(self) -> None:
        price = american_put_binomial(spot=40.0, strike=40.0, rate=0.06, sigma=0.2, maturity=1.0, n_steps=200)
        european = bs_european_put(spot=40.0, strike=40.0, rate=0.06, sigma=0.2, maturity=1.0)
        self.assertGreater(price, european)

    def test_finite_difference_benchmark_is_reasonable(self) -> None:
        price = american_put_finite_difference(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            n_space_steps=200,
            n_time_steps_per_year=2000,
        )
        european = bs_european_put(spot=40.0, strike=40.0, rate=0.06, sigma=0.2, maturity=1.0)
        self.assertGreater(price, european)

    def test_finite_difference_and_binomial_are_close(self) -> None:
        fd_price = american_put_finite_difference(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            n_space_steps=200,
            n_time_steps_per_year=2000,
        )
        tree_price = american_put_binomial(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            n_steps=500,
        )
        self.assertAlmostEqual(fd_price, tree_price, delta=0.08)

    def test_fd_delta_estimator_returns_metadata(self) -> None:
        config = LSMCConfig(n_paths=2_000, seed=11)
        estimate = estimate_delta_fd(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            config=config,
            bump=0.5,
        )
        self.assertIn("estimate", estimate)
        self.assertIn("runtime_sec", estimate)
        self.assertTrue(estimate["metadata"]["common_random_numbers"])

    def test_finite_difference_delta_has_put_sign(self) -> None:
        delta = american_put_delta_finite_difference(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            n_space_steps=200,
            n_time_steps_per_year=2000,
            bump=0.5,
        )
        self.assertLess(delta, 0.0)

    def test_pathwise_delta_has_put_sign(self) -> None:
        config = LSMCConfig(n_paths=20_000, seed=21)
        estimate = estimate_delta_pathwise(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            config=config,
            seed=21,
        )
        self.assertLess(estimate["estimate"], 0.0)
        self.assertIsNotNone(estimate["std_error"])

    def test_pathwise_delta_is_close_to_binomial_benchmark(self) -> None:
        config = LSMCConfig(n_paths=40_000, seed=42)
        pathwise = estimate_delta_pathwise(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            config=config,
            seed=42,
        )
        benchmark = american_put_delta_finite_difference(
            spot=40.0,
            strike=40.0,
            rate=0.06,
            sigma=0.2,
            maturity=1.0,
            n_space_steps=200,
            n_time_steps_per_year=2000,
            bump=0.5,
        )
        self.assertAlmostEqual(pathwise["estimate"], benchmark, delta=0.08)


if __name__ == "__main__":
    unittest.main()
