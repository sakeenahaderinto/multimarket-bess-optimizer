"""
sanity_check_error_pool.py — Phase 2 sanity check.

Builds the historical error-path pool in isolation and inspects it before
plugging make_historical_error_scenario_builder into the Dec 2022 fixed-
month comparison. Checks, in order:

  1. Pool size — how many usable days survived the exclusion filter.
  2. Error magnitude sanity — DA/BM errors should be tens of £/MWh, not
     thousands; DC errors should be small (DC Low/High trade at much lower
     absolute prices than DA/BM, per the calibration findings in Session 1).
  3. A visual spot-check of a few sampled error paths against zero, to
     confirm they look like plausible day-shaped forecast error series
     rather than degenerate (all-zero, all-identical, or wildly erratic).
  4. One scenario built from the pool for a real window, compared against
     the q50-only path it was added to, to confirm the shapes/units line
     up before this touches the optimiser at all.

Usage:
    uv run phase2/sanity_check_error_pool.py
"""

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from run_backtest import _load_actual, _load_dc_actual, _load_forecast

from historical_error_scenarios import build_error_path_pool, make_historical_error_scenario_builder

PERIOD = "cv"  # OOF forecasts cover the CV period (2022-2024)


def main() -> None:
    # ------------------------------------------------------------------
    # Load actuals, normalised the same way every other script in this
    # project does
    # ------------------------------------------------------------------
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual")

    dc_low_raw  = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_30  = dc_low_raw.resample("30min").ffill().to_frame("value")
    dc_high_30 = dc_high_raw.resample("30min").ffill().to_frame("value")

    # ------------------------------------------------------------------
    # Load OOF forecasts
    # ------------------------------------------------------------------
    da_oof  = _load_forecast("da",      PERIOD)
    bm_oof  = _load_forecast("bm",      PERIOD)
    dcl_oof = _load_forecast("dc_low",  PERIOD)
    dch_oof = _load_forecast("dc_high", PERIOD)

    # ------------------------------------------------------------------
    # 1. Build the pool
    # ------------------------------------------------------------------
    pool = build_error_path_pool(
        da_actual_30, bm_actual, dc_low_30, dc_high_30,
        da_oof, bm_oof, dcl_oof, dch_oof,
    )

    n_pool = len(pool["dates"])
    print(f"\n{'='*60}")
    print(f"  1. POOL SIZE: {n_pool} usable historical days")
    print(f"     date range: {pool['dates'].min()} to {pool['dates'].max()}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Error magnitude sanity
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  2. ERROR MAGNITUDE SANITY (£/MWh equivalent, abs values)")
    print(f"{'='*60}")
    for market in ["da", "bm", "dc_low", "dc_high"]:
        vals = pool[market]
        print(
            f"  {market:8s}  mean_abs={np.abs(vals).mean():8.2f}  "
            f"std={vals.std():8.2f}  min={vals.min():9.2f}  max={vals.max():9.2f}"
        )
    print(
        "\n  Expect: da/bm mean_abs in the tens (consistent with the "
        "MAE=14.4 (DA) / MAE=30.8 (BM) test metrics from findings.md "
        "Section 2). dc_low/dc_high mean_abs should be small (consistent "
        "with MAE=2.0 / MAE=1.2 respectively). If any of these are off by "
        "an order of magnitude or more, check unit/column mismatches before "
        "proceeding."
    )

    # ------------------------------------------------------------------
    # 3. Spot-check a few sampled error paths
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  3. SPOT-CHECK: 3 random historical DA error paths")
    print(f"{'='*60}")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_pool, size=min(3, n_pool), replace=False)
    for i in sample_idx:
        date = pool["dates"][i]
        path = pool["da"][i]
        print(f"\n  Date: {date}")
        print(f"  DA error path (48 periods): "
              f"min={path.min():.1f}  max={path.max():.1f}  "
              f"first 6 values: {np.round(path[:6], 1)}")
        if np.allclose(path, 0):
            print("  *** WARNING: this path is all zeros — investigate. ***")
        if np.allclose(path, path[0]):
            print("  *** WARNING: this path is constant — investigate. ***")

    # ------------------------------------------------------------------
    # 4. Build one scenario for a real window and compare against q50
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  4. ONE SCENARIO BUILD, COMPARED AGAINST Q50")
    print(f"{'='*60}")

    test_window_start = pd.Timestamp("2022-12-15", tz="UTC")
    horizon = 96

    da_fc_win  = da_oof.loc[test_window_start : test_window_start + pd.Timedelta(hours=48)].iloc[:horizon]
    bm_fc_win  = bm_oof.loc[test_window_start : test_window_start + pd.Timedelta(hours=48)].iloc[:horizon]
    dcl_fc_win = dcl_oof.loc[test_window_start : test_window_start + pd.Timedelta(hours=48)].iloc[:horizon]
    dch_fc_win = dch_oof.loc[test_window_start : test_window_start + pd.Timedelta(hours=48)].iloc[:horizon]

    builder = make_historical_error_scenario_builder(pool, n=20)
    scenarios = builder(test_window_start, horizon, da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win, seed=42)

    print(f"  Shapes: da={scenarios['da'].shape}, bm={scenarios['bm'].shape}, "
          f"dc_low={scenarios['dc_low'].shape}, dc_high={scenarios['dc_high'].shape}")
    print(f"  Expected: (20, {horizon}) for all four\n")

    compare = pd.DataFrame({
        "q50_da": da_fc_win["q50"].values[:horizon],
        "scenario_0_da": scenarios["da"][0],
        "scenario_5_da": scenarios["da"][5],
        "scenario_10_da": scenarios["da"][10],
    })
    print(compare.head(12).to_string())
    print(
        "\n  Expect: scenario columns should track q50's shape but deviate "
        "by realistic amounts in either direction, with different scenarios "
        "diverging from each other (since each draws a different historical "
        "error day) — not identical to each other, and not identical to q50."
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()