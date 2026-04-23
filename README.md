# American Option Greeks via Pathwise LSMC

This repository contains a notebook-first study of American option pricing and delta estimation with Longstaff-Schwartz Monte Carlo (LSMC), focused on numerical accuracy, validation, and computational efficiency.

## Project Goal

We study a 1D American put under GBM and build the project in four parts:

1. pricing baseline with LSMC,
2. delta baselines with finite difference and pathwise estimation,
3. one technical extension beyond the core delta study,
4. a unified numerical comparison across estimators.

The current repository completes the core Part 1 and Part 2 scope. Parts 3 and 4 are planned next and are included below as the current project roadmap.

## Reference Paper

The main paper currently cited by the project is:

- Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach*. *The Review of Financial Studies, 14*(1), 113-147. [https://doi.org/10.1093/rfs/14.1.113](https://doi.org/10.1093/rfs/14.1.113)

This paper anchors the pricing side of the project:

- American put pricing under GBM,
- least-squares Monte Carlo valuation,
- regression-based continuation estimation,
- weighted Laguerre basis functions,
- replication against the published benchmark setup.

## Implemented Scope

The current implemented scope is:

- LSMC pricing for an American put,
- LS2001 pricing replication,
- finite-difference delta with common random numbers,
- fixed-policy pathwise delta,
- validation against binomial and finite-difference PDE benchmarks,
- robustness checks for path count, bump size, and regression basis.

## Repository Layout

### Current Implemented Layout

```text
src/lsmc_greeks/
  models.py
  payoffs.py
  pricer.py
  utils.py
  greeks/
    finite_diff.py
    pathwise.py
  benchmarks/
    binomial.py
    finite_difference.py

notebooks/
  01_ls2001_replication.ipynb
  02_lsmc_baseline_and_fd_delta.ipynb
  03_pathwise_delta.ipynb

assets/figures/
  ls2001_replication_ee_comparison.png
  finite_difference_delta_baseline.png
  finite_difference_delta_spot_sweep.png
  finite_difference_delta_bump_sensitivity.png
  pathwise_delta_baseline_comparison.png
  pathwise_delta_path_count.png
  pathwise_delta_bump_sensitivity.png
  pathwise_delta_basis_sensitivity.png

tests/
  test_lsmc_baseline.py
```

### Planned Notebook Roadmap

The current repo implements `01` through `03`. The next planned notebooks under the updated project roadmap are:

```text
notebooks/
  01_ls2001_replication.ipynb
  02_lsmc_baseline_and_fd_delta.ipynb
  03_pathwise_delta.ipynb
  04_basis_and_gamma_extension.ipynb or 04_regression_sensitivity.ipynb
  05_lr_or_mixed_extension.ipynb
  06_estimator_comparison.ipynb
```

## Environment

A minimal Conda environment is provided in `environment.yml`.

Core dependencies:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `notebook`
- `ipykernel`

## Methods

### Pricing

The pricing baseline uses Longstaff-Schwartz Monte Carlo for a Bermudan approximation to the American put. The continuation regression is built from weighted Laguerre basis functions, and the implementation supports basis-sensitivity experiments through the `basis_degree` parameter.

### Delta Estimators

Two LSMC delta estimators are currently implemented:

- `src/lsmc_greeks/greeks/finite_diff.py`: central bump-and-revalue with common random numbers,
- `src/lsmc_greeks/greeks/pathwise.py`: fixed-policy pathwise delta, where the learned LSMC stopping rule is held fixed during differentiation.

### Validation Benchmarks

Two independent benchmark layers are used:

- `src/lsmc_greeks/benchmarks/binomial.py`,
- `src/lsmc_greeks/benchmarks/finite_difference.py`.

This reduces dependence on any single reference method.

### Fixed-Policy Interpretation

The pathwise estimator should be interpreted carefully. The American option value is an optimal-stopping problem, while the implemented pathwise estimator differentiates the discounted payoff under the learned LSMC stopping rule and treats that stopping rule as fixed during differentiation.

Benchmark agreement supports this estimator empirically, but it does not by itself prove that the implementation is computing the exact derivative of the true optimal-stopping value. In the report, this method should be framed as a validated first-order estimator under a fixed-policy assumption.

## Project Roadmap

### Part 1: Pricing Baseline

Goal:

- build and validate the American put LSMC pricing engine.

Current status:

- complete.

Included work:

- GBM simulation,
- payoff logic,
- LSMC pricing,
- LS2001 replication,
- benchmark setup foundation.

### Part 2: Delta Baselines

Goal:

- build and validate the core delta estimators.

Current status:

- complete for the agreed core scope.

Included work:

- finite-difference delta with common random numbers,
- pathwise delta under a fixed-policy assumption,
- validation against binomial and PDE benchmarks,
- initial robustness checks.

### Part 3: Advanced Extension

Goal:

- add one technical extension beyond the core delta study.

Planned examples:

- likelihood-ratio or mixed estimator,
- gamma-oriented extension,
- deeper regression or basis study.

Current status:

- not started.

### Part 4: Unified Numerical Comparison

Goal:

- integrate the estimators into one technical benchmark study.

Planned outputs:

- repeated-seed comparison tables,
- runtime versus accuracy comparison,
- unified estimator comparison notebook.

Current status:

- not started.

## Planned Work Distribution

This is the current planned split for the next phase of the project. Replace the generic labels with teammate names before final submission if desired.

### Person 1: Pricing Infrastructure Lead

Owns:

- `src/lsmc_greeks/models.py`
- `src/lsmc_greeks/payoffs.py`
- `src/lsmc_greeks/utils.py`
- `notebooks/01_ls2001_replication.ipynb`

### Person 2: LSMC Pricer + Basis/Regression Lead

Owns:

- `src/lsmc_greeks/pricer.py`
- the basis and regression logic inside the pricer,
- the future Part 3 basis/regression extension notebook, suggested as `04_basis_and_gamma_extension.ipynb` or `04_regression_sensitivity.ipynb`.

### Person 3: Finite-Difference Delta + Extension Lead

Owns:

- `src/lsmc_greeks/greeks/finite_diff.py`
- `notebooks/02_lsmc_baseline_and_fd_delta.ipynb`
- the future Part 3 extension estimator notebook, suggested as `05_lr_or_mixed_extension.ipynb`.

### Person 4: Pathwise Delta + Benchmarks + Comparison Lead

Owns:

- `src/lsmc_greeks/greeks/pathwise.py`
- `src/lsmc_greeks/benchmarks/binomial.py`
- `src/lsmc_greeks/benchmarks/finite_difference.py`
- `notebooks/03_pathwise_delta.ipynb`
- the future unified comparison notebook, suggested as `06_estimator_comparison.ipynb`.

## Notebook Storyline

The canonical notebook sequence is:

1. `notebooks/01_ls2001_replication.ipynb`: validate the pricing baseline against LS2001.
2. `notebooks/02_lsmc_baseline_and_fd_delta.ipynb`: establish the finite-difference delta baseline.
3. `notebooks/03_pathwise_delta.ipynb`: compare pathwise delta against the baseline and the external benchmarks.

Older draft notebooks outside the `notebooks/` directory are not part of the current structure.

## Results to Date

### Pricing Baseline

The pricing layer was first validated against the LS2001 American put benchmark.

Summary diagnostics from `notebooks/01_ls2001_replication.ipynb`:

- mean absolute early-exercise-premium difference: `0.010229`
- max absolute early-exercise-premium difference: `0.036640`
- rows with `|diff| <= 0.01`: `12/20`

![LS2001 replication](assets/figures/ls2001_replication_ee_comparison.png)

### Finite-Difference Delta Baseline

At the baseline case `S=40, K=40, r=0.06, sigma=0.20, T=1`:

- LSMC price: `2.314678`
- LSMC finite-difference delta: `-0.412321`
- binomial benchmark delta: `-0.404959`
- PDE benchmark delta: `-0.405150`
- absolute error versus binomial: `0.007363`
- absolute error versus PDE: `0.007171`
- runtime: `0.189245` seconds

![Finite-difference baseline](assets/figures/finite_difference_delta_baseline.png)

Across the spot sweep `S=36, 38, 40, 42, 44`, the finite-difference estimator tracks the benchmark curves closely, with the largest absolute error versus the binomial benchmark around `0.010215` at `S=38`.

![Finite-difference spot sweep](assets/figures/finite_difference_delta_spot_sweep.png)

The bump sweep shows the main practical weakness of bump-and-revalue: the estimate changes materially when the bump is very small.

![Finite-difference bump sensitivity](assets/figures/finite_difference_delta_bump_sensitivity.png)

### Baseline Delta Comparison

At the same baseline case, the pathwise estimator is closer to the binomial benchmark than the LSMC finite-difference baseline in the current run.

| Method | Delta | Std. Error | Runtime (s) | Abs. Error vs Binomial |
| --- | ---: | ---: | ---: | ---: |
| LSMC finite difference | -0.412321 |  | 0.196072 | 0.007363 |
| LSMC pathwise | -0.400928 | 0.002106 | 0.103443 | 0.004031 |
| Binomial benchmark | -0.404959 |  |  | 0.000000 |
| Finite-difference benchmark | -0.405150 |  |  | 0.000191 |

![Baseline delta comparison](assets/figures/pathwise_delta_baseline_comparison.png)

This is a benchmarked result for the current configuration, not a claim that pathwise is always better.

### Robustness Findings

#### Path Count

Using repeated seeds over path counts `5,000`, `10,000`, `20,000`, and `40,000`, the pathwise estimator remains competitive while running faster than bump-and-revalue in the current implementation.

Selected repeated-seed mean absolute errors versus the binomial benchmark:

- `5,000` paths: FD `0.007654`, pathwise `0.007982`
- `10,000` paths: FD `0.011839`, pathwise `0.008359`
- `20,000` paths: FD `0.006979`, pathwise `0.004351`
- `40,000` paths: FD `0.003269`, pathwise `0.005014`

The figure below should be read as a repeated-seed summary with variability bars, not as a claim of exact monotone convergence.

![Path-count robustness](assets/figures/pathwise_delta_path_count.png)

#### Bump Size

At the baseline configuration with `20,000` paths, the pathwise estimate stays fixed at `-0.399138`, while the finite-difference estimate varies from `-0.425612` at a `0.10` bump to roughly `-0.4122` for larger bumps.

![Bump-size sensitivity](assets/figures/pathwise_delta_bump_sensitivity.png)

#### Basis Sensitivity

In the current benchmark configuration, the LSMC price and both delta estimators move as the regression basis changes. This matters because the pathwise estimator inherits model risk from the learned stopping rule, not just Monte Carlo noise.

At basis degrees `0, 1, 2, 3`, the pathwise absolute error versus the binomial benchmark is approximately:

- `0.020502`
- `0.006442`
- `0.005821`
- `0.002846`

The figure is best read as a sensitivity check for this setup, not as a universal ranking of basis choices.

![Basis sensitivity](assets/figures/pathwise_delta_basis_sensitivity.png)

## Interpretation

The current project takeaway is that pathwise delta is promising because it:

- avoids the extra tuning parameter required by bump-and-revalue,
- compares reasonably well with independent numerical benchmarks,
- and remains competitive across the current robustness sweeps.

The main caveat is also central to the project:

- the current pathwise estimator treats the learned LSMC stopping rule as fixed during differentiation.

For the report, this should be described as a validated first-order estimator for delta, not as a complete treatment of the American exercise-boundary derivative.

## Reproducibility

- Core implementation lives in `src/`.
- Notebook analysis and presentation live in `notebooks/`.
- Report-ready figures are generated into `assets/figures/`.
- Random-number generation is explicit so later estimators can share common random numbers.
- Validation uses both binomial and finite-difference benchmark solvers.

## Testing

The current baseline includes tests for:

- GBM path generation,
- payoff logic,
- LS2001 pricing sanity checks,
- binomial and finite-difference benchmark behavior,
- finite-difference estimator metadata,
- pathwise delta sign and benchmark consistency,
- Laguerre basis-size behavior.

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

## Next Steps

- begin Part 3 with one technical extension,
- add a unified estimator comparison notebook for Part 4,
- replace generic person labels with teammate names in the work distribution section,
- extend the final submission materials with team contribution notes.
