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

- `da_charge`, `da_discharge`: Day-ahead schedule (MW) --- *First-stage (scenario-independent)*
- `dc_low`, `dc_high`: DC capacity per EFA block (MW) --- *First-stage (scenario-independent)*
- `bm_offer`: BM offer capacity submitted to ESO (MW) --- *First-stage (scenario-independent)*
- `bm_dispatch`: BM accepted volume in scenario $s$ --- *Second-stage recourse (scenario-dependent)*
- `soc`: State of charge in scenario $s$ (fraction) --- *Second-stage recourse (scenario-dependent)*
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

**BM simplification.** BM offer is committed as a first-stage decision, with second-stage scenario recourse $(bm_dispatch \le bm_offer)$. realized revenue settles against the system sell price. In GB market operations, ESO dispatches units selectively via Bid-Offer Acceptances (BOAs). We assume full acceptance of offered capacity, and this serves as an optimistic revenue proxy that may overstate BM revenues.

**Degradation convention.** Degradation cost uses `replacement_cost / (2 × cycle_life)` applied to both charge and discharge throughput, treating one full cycle as one charge plus one discharge event.

---

