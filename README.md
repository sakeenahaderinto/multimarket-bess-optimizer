# Multimarket BESS Optimizer

## Overview

The Multimarket BESS Optimizer is a pre-market backtesting tool for optimizing Battery Energy Storage Systems (BESS) across multiple energy markets in Great Britain. This repository demonstrates an end-to-end pipeline, moving from raw data ingestion and feature engineering to probabilistic price forecasting and scenario-based Mixed-Integer Programming (MIP).

The challenge in battery optimization is that multiple markets compete for the same physical asset. This model simultaneously optimizes dispatch across three distinct revenue streams:

- **Day-Ahead (DA) Market:** Wholesale energy arbitrage.
- **Dynamic Containment (DC):** Frequency response ancillary services (DC Low and DC High).
- **Balancing Mechanism (BM):** Real-time grid balancing.

By leveraging quantile regression and stochastic optimization, the model finds the dispatch allocation that maximizes expected net revenue, accounting for degradation costs across a distribution of possible price scenarios.

---
## Architecture

```mermaid
flowchart LR
A[Data Ingestion] --> B[Feature Engineering]
B --> C[Probabilistic Forecasting]
C --> D[Scenario Generation]
D --> E[Pyomo MIP Optimiser]
E --> F[Walk-Forward Backtest]
```

1. **Ingestion**: Retrieves raw historical market data from Elexon BMRS (system prices), Electricity Maps (N2EX day-ahead prices), and NESO (DC auction results).
2. **Feature Pipeline**: Merges and aligns the data sources at a 30-minute resolution. Generates cyclical time encodings, rolling statistics, and safely lags real-time market data to strictly prevent target leakage and look-ahead bias.
3. **Forecasting**: Trains LightGBM quantile regression models ($q_{10}$, $q_{50}$, $q_{90}$) using walk-forward cross-validation to predict DA, BM, and DC clearing prices.
4. **Scenario Generation**: A Gaussian copula with Cholesky decomposition samples correlated price scenarios across all four markets, preserving cross-market dependence structure estimated from historical data.
5. **Optimisation**: A Pyomo-based MIP that maximizes expected net revenue across generated price scenarios over a 48-hour horizon. Solved using the HiGHS solver.
6. **Backtesting**: Walks forward day-by-day, solving the MIP using out-of-sample forecast scenarios and settling revenue against actual, realized market prices to accurately evaluate strategy performance.

---
## Key Features

- **Stochastic Optimization**: Translates quantile forecasts into a discrete fan of correlated price scenarios, allowing the MIP to make robust dispatch decisions under market uncertainty.
- **Constraint Modeling**: Handles complex operational constraints, including asymmetric round-trip efficiency ($\sqrt{RTE}$ applied per side), mutually exclusive charge/discharge modes (via binary variables), and energy headroom reservations required to fulfill Dynamic Containment availability.
- **Leakage-Free Validation**: Enforces a strict 48-period (24-hour) minimum lag for all real-time market features to perfectly align with day-ahead gate closures, ensuring a temporally consistent backtest.
- **Baseline Benchmarking**: Compares the full probabilistic model against three deterministic baselines (median forecast, previous-day naive, previous-week naive) using a shared clean-date evaluation set for apples-to-apples comparison.

---
## Markets Modelled

| Market                       | Revenue Mechanism                                           | Commitment Timing                                                                  |
| ---------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Day-Ahead (DA)**           | Buy/sell against N2EX prices                                | Fixed schedule before gate closure                                                 |
| **Dynamic Containment (DC)** | Availability payment per MW reserved for frequency response | Committed per 4-hour EFA block in daily auction                                    |
| **Balancing Mechanism (BM)** | Accepted offer price when ESO dispatches the unit           | Opportunistic — no advance commitment, assumes 100% acceptance as a simplification |

---
## Optimisation Model

**Decision Variables:**

- `da_charge`, `da_discharge`: Day-ahead schedule (MW) --- *Scenario-independent (committed before uncertainty resolves)*
- `dc_low`, `dc_high`: DC capacity per EFA block (MW) --- *Scenario-independent*
- `bm_offer`: BM dispatch (MW) --- *Scenario-dependent (decided after uncertainty resolves)*
- `soc`: State of charge (fraction) --- *Scenario-dependent*
- `charge_mode`: Binary variable preventing simultaneous DA charging and discharging.

**Objective Function:**

Maximize the expected net revenue across all scenarios minus the expected degradation cost, using a 0.5h duration factor to convert MW to MWh.

**Key Constraints:**

- **SoC Dynamics:** Energy balance equations with asymmetric round-trip efficiency tracking.
- **DC EFA Block Consistency:** DC commitments must remain constant across each 4-hour EFA block.
- **DC Energy Headroom:** The battery must hold sufficient SoC headroom to physically deliver the committed DC response if called upon.
- **Terminal SoC bounds:** Ensures the battery is not fully depleted at the end of the optimization horizon.

For the full mathematical formulation see [`experiments/optimisation_model_formulation.md`](experiments/optimisation_model_formulation.md).

---
## Quickstart

**1. Install Dependencies**

```bash
git clone https://github.com/sakeenahaderinto/multimarket-bess-optimizer
cd multimarket-bess-optimizer
uv sync
```

**2. Pull Historical Data**

```bash
uv run backfill.py
```

**3. Train Forecasters**

Builds the feature table and trains all four quantile models (DA, BM, DC Low, DC High). Saves out-of-fold predictions for CV evaluation and out-of-sample predictions for 2025 evaluation.

```bash
uv run train_forecasters.py
```

**4. Run Baseline Comparison**

Evaluates four strategies — the full copula model, a deterministic median baseline, and two naive lag baselines — against the same set of dates.

```bash
uv run run_baselines.py --period cv   # 2022-2024 out-of-fold evaluation
uv run run_baselines.py --period oos  # 2025 out-of-sample evaluation
```

Results are saved to `data/results/baseline_comparison.csv`.

**5. Run Backtest (single strategy)**

```bash
uv run run_backtest.py --period cv
uv run run_backtest.py --period oos
```

Optional arguments:
- `--start` / `--end`: Override date range within the period
- `--step N`: Step by N days between backtest windows (default: 1)

> **Note:** The forecast files (`{model_id}_oof_2022_2024.parquet` and `{model_id}_oos_2025.parquet`) are generated by `train_forecasters.py` and represent historical predictions for backtesting purposes only. They are not live, operational next-day forecasts.

---
## Project Structure

```
multimarket-bess-optimizer/
├── optimiser/           # Pyomo MIP model and scenario generation
├── forecasting/         # LightGBM quantile forecasters
├── features/            # Feature engineering and lag alignment pipeline
├── ingestion/           # Data fetching scripts for BMRS, Electricity Maps, NESO, and weather
├── backtest/            # Walk-forward backtesting engine and revenue settlement
├── baselines/           # Deterministic baseline scenario builders
├── experiments/         # Experiment logs and mathematical formulation
├── notebooks/           # Exploratory data analysis
├── train_forecasters.py # Train all forecasting models
├── run_baselines.py     # Four-strategy baseline comparison
├── run_backtest.py      # Single-strategy walk-forward backtest
├── backfill.py          # Historical data ingestion
└── data/                # Raw data, features, and forecasts (gitignored)
```

---
## Scripts Reference

**Pipeline scripts**

| Script | Description |
|---|---|
| `backfill.py` | Pulls historical raw data from Elexon BMRS, Electricity Maps, NESO, and Open-Meteo. Run once to populate `data/raw/` before training. |
| `data_quality.py` | DA price imputation audit. Identifies every half-hourly slot where the DA price was forward-filled by more than one step and classifies it by severity of use (settlement window, training target, or lag feature). Writes `data/quality/da_imputed_slots.parquet` and `data/quality/da_imputed_by_date.parquet`. |
| `train_forecasters.py` | Builds the feature table, trains all four LightGBM quantile models (DA, BM, DC Low, DC High) using walk-forward CV, and saves OOF predictions for 2022–2024 and final-model OOS predictions for 2025 to `data/forecasts/`. |
| `train_da_hourly.py` | Trains an alternative DA forecaster at native hourly resolution (one prediction per hour, then upsampled by duplication to 30-min). Built to test whether the spurious sub-hourly forecast variance in the 30-min DA model affects arbitrage revenue. |
| `run_baselines.py` | Runs four strategies — the full Gaussian copula model, a deterministic median-forecast baseline, and two naive lag baselines (previous-day, previous-week) — against the same set of valid dates, and saves a side-by-side revenue comparison to `data/results/baseline_comparison.csv`. |
| `run_backtest.py` | Runs the full stochastic optimiser in walk-forward mode for a single period (CV or OOS), settling each day's decisions against realised market prices, and saves per-day revenue breakdown to `data/backtest/`. |
| `run_fixed_month_comparison.py` | Compares model performance within a fixed calendar month across strategies, useful for isolating regime-specific behaviour (e.g. high-volatility winter periods). |

**Diagnostics and experiment scripts**

| Script | Description |
|---|---|
| `historical_error_scenarios.py` | Implements the historical error-path scenario generator. Builds a pool of complete joint forecast-error days (DA, BM, DC Low, DC High) and returns a `scenario_builder` callable that samples from this pool, preserving temporal and cross-market dependence by construction rather than via an estimated correlation matrix. |
| `check_q50_spread_sign.py` | Checks whether the BM-DA q50 spread sign mismatch is a systematic, volatility-conditional failure or an isolated event. Computes daily spread sign accuracy across the CV period, splits by volatility quartile, and supports swapping in the hourly DA forecaster for comparison. |
| `check_pool_spike_magnitude.py` | Compares the error pool's typical error magnitude at the 16:00–19:00 spike window against the realised spread on a specific target day, to test whether universal passivity in the copula strategy is explained by the pool lacking days with large enough error magnitudes. |
| `sanity_check_error_pool.py` | Pre-flight sanity check for the historical error-path pool before connecting it to the optimiser. Checks pool size, error magnitude plausibility, visual shape of sampled paths, and one full scenario construction. |
| `inspect_day_schedule.py` | Extracts and prints the per-period dispatch schedule (DA charge/discharge, BM offer, DC capacity, SOC) for a single backtest day. Mirrors `run_backtest`'s single-window logic but keeps the solved Pyomo model to read decision variables directly. |
| `check_hourly_lag_alignment.py` | Manual sanity check of the hourly DA feature pipeline's lag construction before a full training run, covering lag correctness, demand volume summing, and a manual check of `spread_lag_24` (which is skipped by `base.py`'s automatic lag loop). |
| `excluded_dates.py` | Registry of dates excluded from the historical error-path sampling pool: UK clock-change dates (2022–2025) and the six genuine BM data-gap anomaly days. |

---
## Known Limitations

**Training data coverage.** The model is trained on 2022–2024 data. DC auction data prior to 2022 has incomplete market coverage as the service was still being rolled out, so 2022 is used as the study start. The CV backtest (2022–2024) uses out-of-fold predictions to eliminate look-ahead bias. The OOS backtest (2025) evaluates true unseen data, but market regime changes in 2025 that differ substantially from the training period may degrade forecast accuracy.


**DC price resolution.** Dynamic Containment prices clear per 4-hour EFA block, not per 30-minute settlement period. The pipeline forward-fills block-level clearing prices to 30-minute resolution for consistency with the optimisation model. Forecast accuracy metrics for DC should be interpreted at the block level, not the half-hourly level.

**DA forecast framing and decision timeline.** This is a pre-market backtesting tool. The battery optimizes its day-ahead schedule based on price forecasts available before gate closure (11:00 UTC D-1 for GB day-ahead market). All features use a minimum 48-period (24-hour) lag to respect this constraint.

**Single asset, no network constraints.** The model treats the battery as connected to an infinite bus with no feeder capacity constraints or topology awareness. In practice, a battery's optimal dispatch depends on its location in the distribution network.

**BM simplification.** BM offer is modelled as a scenario-dependent dispatch settled at the system sell price. A production model would distinguish offers from bids, model the baseline physical notification, and account for bid/offer acceptance probability explicitly.

**Degradation convention.** Degradation cost uses `replacement_cost / (2 × cycle_life)` applied to both charge and discharge throughput, treating one full cycle as one charge plus one discharge event.

---
## Changelog

A log of significant decisions and fixes across development sessions. Full detail in [`session_fixes_and_decisions.md`](session_fixes_and_decisions.md).

### 2026-06-12: Session 1

- **BM settlement overstated 20×** — BM revenue divided by number of scenarios, not summed across them.
- **SOC carryover silent failure** — `_read_ending_soc` tried to unpack 3-index Pyomo variable as 2-index; SOC was never carried forward, every backtest day started at 50%. Fixing this reduced annualised revenue from ~£72k (inflated) to ~£52k (correct).
- **Terminal SOC constraint** — was forcing the battery back to starting SOC by end-of-horizon regardless of economic cost. Replaced with a configurable 10% floor.
- **Double-counting fix** — backtest was settling all 96 horizon periods; added `SETTLE_PERIODS = 48` to settle only the first 24 hours per step.
- **ThreadPoolExecutor removed** — LightGBM's internal thread pool makes Python thread wrapping counterproductive.
- **Lag alignment check fixed** — `_check_lag_alignment` was comparing each lag column against the model target rather than its own source column; replaced correlation test with exact-shift residual test.
- **DA price ffill limit** — extended from `limit=4` (2h) to `limit=96` (48h); collapsed OOF gap from 30,369 missing slots to near zero.
- **Rolling stats `min_periods`** — added `min_periods=window//2` to prevent single NaN rows propagating 335 rows of missing features.

### 2026-06-12: Session 2

- **Explicit study period constants** — `STUDY_START`/`STUDY_END` added to `pipeline.py`; canonical index no longer derived from data bounds.
- **Correlation matrix training cutoff** — `_estimate_correlation_matrix` now filtered to `cutoff_date` (default: 2025-01-01) to prevent leaking future dependence into the scenario generator.
- **Spearman-to-Gaussian conversion** — applied `ρ_G = 2·sin(π·ρ_S/6)` before Cholesky decomposition.
- **DC EFA block snapping** — DC scenario draws now held constant within each 4-hour EFA block (matching market structure); initial implementation used block-mean, later corrected (see Session 3).
- **Data-length check key collision** — `{**actuals, **forecasts}` with duplicate keys caused forecast windows to silently overwrite actuals in the length check. Renamed keys to `_actual`/`_fc` suffixes.

### 2026-06-14: Session 3

- **DC block snapping corrected** — replaced block-mean uniform draw with first-period draw (`::8`), preserving full U(0,1) spread and Cholesky-encoded cross-market correlation.
- **DC quantile forecasts block-averaged** — `q10`/`q50`/`q90` for DC markets averaged to EFA-block level before inverse QF, so scenario prices are block-constant end-to-end.
- **Correlation matrix per backtest step** — moved `_estimate_correlation_matrix` inside the backtest loop, passing `current_date` as the cutoff to prevent future data leakage in CV steps.
- **Lag check extended** — generalised from `_lag_48` only to all `_lag_N` columns via regex; all four lag lengths (48, 96, 144, 336) now validated.
- **`data_diagnostics.ipynb` created** — DA gap distribution, imputation vs frequency-alignment classification, series coverage, feature NaN audit.

### 2026-06-15: Session 4

- **`data_quality.py` created** — slot-level DA imputation audit with three severity contexts (settlement, training target, lag feature); writes joinable flag to backtest results.
- **Imputed-day flag in backtest results** — `da_imputed` column added to results DataFrame for paper reporting of clean vs imputed day performance.
- **Eigenvalue regularisation threshold** — changed from `< 0` to `< 1e-8`; catches near-singular matrices before Cholesky fails.
- **`step_days != 1` guard** — explicit error if non-daily step size passed; `SETTLE_PERIODS = 48` assumes daily steps.
- **SOC boundary unit test** — `test_soc_boundary_at_settle_t` added to confirm `_read_ending_soc(model, id, settle_t=N)` returns exactly `soc[N]` averaged across scenarios.

### 2026-06-17: Session 5

- **DA settlement temporal misalignment bug** — `_normalise_to_30min` was called for DC actuals before `settle_revenue` but never for `da_actual`. With hourly data filling the settled array, period `t` was indexed against approximately twice the elapsed real time. Every strategy was affected; fixing it increased net revenue across all strategies (40% for `main`, 18% for `median`, 156% for `prev_day`). Residual `main` vs `median` gap narrowed from 65% to 40%.
- **`processed/` folder introduced** — raw multi-file sources concatenated (no transformation) into single files in `data/processed/`. Required path fixes in three independent files: `pipeline.py`, `run_backtest.py`, `run_baselines.py`, and `scenarios.py` (where a leftover plural-variable bug from the old multi-file load pattern also needed fixing).
- **Study period changed to 2022** — DC auction data for 2021 has incomplete market coverage. `STUDY_START` changed to `"2022-01-01"` throughout.

### 2026-06-25 — Session 6

- **December 2022 fixed-month comparison** — perfect foresight establishes the upper bound (£13,069 net, 31 days). `median` (£3,740) outperforms `current_scenarios` (£2,254). Perfect foresight's negative DA revenue confirmed as intentional DA→BM spread arbitrage.
- **DA–BM correlation finding** — empirically estimated correlation (0.641) substantially weaker than the hardcoded fallback (0.85), and likely averaged over calm and volatile regimes. On spike days the relationship may be much stronger, but the copula generates scenarios where the co-movement is unreliable, leading the optimizer to rational passivity.
- **Historical error-path scenario generator** (`historical_error_scenarios.py`) — implemented `make_historical_error_scenario_builder`. Samples complete joint historical forecast-error days across all four markets; preserves both temporal and cross-market dependence by construction. `excluded_dates.py` created as the canonical registry of dates unsuitable for error-path sampling.
- **Hourly DA forecaster** (`train_da_hourly.py`, `features/pipeline_da_hourly.py`) — separate hourly-resolution pipeline eliminates spurious `:00` vs `:30` forecast differences. Treated as a comparison arm via `--use-hourly-da` flag, not a replacement for the main DA model.

### 2026-07-01 — Session 7

- **Temporal leakage fix in `historical_error_scenarios.py`** — error pool was filtered once at factory time; backtest steps early in the CV period could sample from future error days. Fixed by filtering to `pool_dates < window_start.date()` inside the builder on every call, with a clear error if too few days are available.
- **Magnitude-weighted spread sign accuracy** — `check_q50_spread_sign.py` extended with `mw_accuracy_pct` (weighted by |actual_spread|) alongside raw mismatch rate, in both the summary printout and the volatility-quartile table. Reveals December 2022 was 49.1% MW accuracy using implied spread.
- **Direct BM-DA spread forecaster** (`forecasting/bm_da_spread.py`, `train_spread.py`) — `BMDASpreadForecaster` trains directly on `bm_price - da_price` as the target, eliminating compounded marginal quantile errors. OOS MAE £26.98, 88.3% coverage. December 2022 MW sign accuracy: 49.1% (implied) → 73.7% (direct spread). Full CV improvement modest (+2.1pp). `--use-spread-model` flag added to `check_q50_spread_sign.py`.
- **Spread model integrated into scenario construction** — `sample_scenarios_multimarket` extended with `spread_fc` parameter; `run_backtest.py` gets `--use-spread-model` flag; `run_fixed_month_comparison.py` made configurable via `--start`/`--end`/`--label`/`--period` and gains a fifth `spread_scenarios` strategy. Plumbing wired through `backtest/engine.py`.
- **Three-regime fixed-month comparison** — Dec 2022 (volatile), Apr 2022 (mixed), May 2024 (calm). `median` wins in every regime. Initial additive spread integration showed variance amplification (BM variance = DA variance + spread variance), causing over-commitment to DA-BM arbitrage in volatile months. `historical_error_scenarios` catastrophic in calm May 2024 (−£1,078): error pool dominated by energy-crisis days creates unrealistic scenarios.
- **q50-shift spread refinement** — replaced additive approach with center-only correction: `bm_q50_shifted = da_fc["q50"] + spread_fc["q50"]`, retaining original BM interval half-widths. Dec 2022 improved from £839 → £1,797; May 2024 now tied with `current_scenarios`. April 2022 introduced 3 solve failures (not yet investigated). `median` still wins all three regimes

### 2026-07-05 — Session 8

- **Spread quantile recalibration** (`calibrate_spread.py`) — post-hoc binary-search calibration finds scaling factor s=0.9490 that achieves exactly 80% OOF coverage (was 82.8%); overwrites both OOF and OOS spread forecast files in place. q50 unchanged; sign accuracy unchanged.
- **True additive spread construction** (`optimiser/scenarios.py`) — replaced q50-shift with direct use of spread model quantiles as BM copula inputs plus post-loop `scenarios["bm"] = scenarios["da"] + scenarios["bm"]`. Calibrated intervals eliminate the 3 April 2022 solve failures introduced in Session 7. April 2022 now beats `current_scenarios` (£4,143); December 2022 fell vs q50-shift (£1,008) because the 5.1% interval shrink is insufficient to tame variance amplification at DA std=152.
- **Regime-conditioned error-path sampling** (`historical_error_scenarios.py`) — uniform pool sampling replaced with volatility similarity × recency weighting. Added `_compute_da_vol_7d` helper, `da_vol_7d` stored in pool dict, `da_actual`/`recency_halflife` parameters to factory function. Weights: `exp(−|pool_vol − target_vol| / bandwidth) × exp(−log(2)/halflife × days_ago)`, default halflife=90 days. May 2024 recovered from −£1,078 to +£48; December 2022 initially got worse (regime conditioning up-weighted the most extreme crisis days).
- **Error magnitude capping** (`historical_error_scenarios.py`) — per-market p95 error caps computed at factory time, applied before adding error paths to q50. Decisive for December 2022 (−£794 with 2 solve failures → £1,913 with 0 failures). No effect on May 2024 (calm-period errors already below cap). Final three-regime results: Dec 2022 £1,913 vs £2,254 `current_scenarios`; Apr 2022 £4,569 (best non-perfect strategy); May 2024 £48 (structural optimizer issue remains).
- **CVaR objective — attempted and reverted** — Rockafellar-Uryasev CVaR formulation added to `optimiser/model.py`; all 31 days failed to solve across all strategies. Both `pyo.Expression` and inline profit implementations tried. Root cause not fully diagnosed; both files restored to pre-CVaR state. Deferred to next session.

### 2026-07-11 — Session 9

- **Rolling spread calibration — leakage fix** (`calibrate_spread.py`) — replaced global s=0.9490 with `compute_rolling_s()`: each date's calibration uses only errors strictly before that date, with s=1.0 for the first 90 days. Output files renamed to `spread_calibrated_oof_2022_2024.parquet` / `spread_calibrated_oos_2025.parquet` to prevent in-place overwrite masking the calibration on rerun. Rolling s: min=0.9490, median=1.0000, max=1.0920. OOF coverage 82.8% → 83.5% (honest). Tail balance diagnostics added.
- **Spread copula correlation basis fix** (`optimiser/scenarios.py`, `backtest/engine.py`) — `_estimate_correlation_matrix` and `_default_correlation_matrix` now take `use_spread: bool` parameter; when `True`, the aligned DataFrame uses `"spread": hist_bm − hist_da` in slot 2 instead of `"bm"`. Root cause: `backtest/engine.py` pre-computed the matrix before `sample_scenarios_multimarket`, bypassing the in-function guard. Fixed by adding `use_spread=spread_fc_win is not None` to the engine call. `bm_quantiles` renamed `second_dim_quantiles`.
- **EFA block alignment fix** (`optimiser/scenarios.py`, `historical_error_scenarios.py`) — new `_efa_block_groups(window_start, horizon)` function: uses `pytz.timezone("Europe/London")` to compute the UTC offset and derives the first real EFA boundary after midnight (period 6 in GMT, period 4 in BST). DC scenario tiling and DC q50 block-averaging now use real EFA groups instead of stride-8 from midnight. Fixed a `NameError` ordering bug in `historical_error_scenarios.py` where `_block_avg_win` was defined after its call. December 2022 `current_scenarios` solve failures: 1 → 0.
- **Error cap/bandwidth leakage fix + ESS guard** (`historical_error_scenarios.py`) — volatility bandwidth and per-market error caps moved from factory time to builder time, computed only from `error_pool["*"][available]` where `available = pool_dates < target_date`. ESS guard added: if `ESS < n_scenarios`, blend regime weights with uniform (`alpha = ESS/n`). Full per-date diagnostics logged (target_vol, bandwidth, ESS, caps, top-5 source dates, year shares). `--error-cap-pct` added to `run_fixed_month_comparison.py` (default 0.95). Cap sensitivity results: p95 is optimal (Dec 2022 £2,221 vs £1,833 uncapped; leakage fix alone worth +£290).
- **Three-level magnitude-weighted sign accuracy** (`check_q50_spread_sign.py`) — `_compute_all_mwsa()` added, reporting MWSA at period (30-min), EFA block (4h), and daily granularity using correct block boundaries. Key finding: daily MWSA (73.7% spread model, Dec 2022) collapses to 61.5% at period level. Full CV period: spread model 0.8pp WORSE at period level (54.6%) despite 2.1pp better at daily level (64.1%). Both models near-random at period level in December 2022.
- **q50 anchor experiment** (`run_fixed_month_comparison.py`) — `make_q50_anchored_builder(base_builder, anchor_count)` replaces the first k of 20 scenarios with the pure q50 path. Strictly monotonic result: more q50 anchors → better revenue in both Dec 2022 (£71.65 → £106.26 → £120.63) and May 2024 (£3.26 → £14.57 → £19.72). Diagnosis: scenario quality problem (period-level MWSA ~55%), not tail-risk problem. Stochastic scenarios add noise that pulls the optimizer toward BM arbitrage the market does not reward.
- **CVaR implementation — working** (`optimiser/model.py`, `run_fixed_month_comparison.py`) — Rockafellar-Uryasev CVaR formulation implemented correctly using a plain Python helper `_scenario_net_rev(s)` (avoids `pyo.Expression` shared-reference failure from Session 8). Conditional on `cvar_lambda < 1.0`; fully backward-compatible. Strategy tuples extended to 4-element `(label, builder, spread_fc, extra_settings_dict)`. Best no-failure CVaR variant (`cvar_l05_a09`, λ=0.5, α=0.9): Dec 2022 £84.23/day, May 2024 £8.67/day. CVaR helps over baseline but less effective than q50 anchoring in the volatile month — adjusting the objective vs adjusting the inputs.
- **Overarching conclusion** — the binding constraint is scenario quality at period level (~55% MWSA), not objective formulation. Next priority: run `spread_scenarios` with all Session 9 fixes applied (rolling calibration, correct EFA blocks, correct correlation basis) to test whether the spread model's December 2022 block-level accuracy (65.6%) translates to optimizer improvement.
