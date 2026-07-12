"""
check_pool_spike_magnitude.py — Compare the historical error pool's typical
magnitude at the 16:00-19:00 window against the actual realised spread on
2022-12-15, to test whether universal passivity (seed-independent, per the
seed-sweep check) is explained by the pool simply not containing days with
error magnitudes large enough to make the optimizer commit confidently.

Usage:
    uv run phase2/check_pool_spike_magnitude.py
"""

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from run_backtest import _load_actual, _load_dc_actual, _load_forecast

from historical_error_scenarios import build_error_path_pool

PERIOD = "cv"
TARGET_DATE = "2022-12-15"

# 16:00-19:00 covers the actual Dec 15 spike window inspected earlier
# (periods 32-37 of the 48-period day, 0-indexed half-hours from 00:00)
SPIKE_PERIOD_START = 32  # 16:00
SPIKE_PERIOD_END   = 39  # 19:30 inclusive


def main() -> None:
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual")

    dc_low_raw  = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_30  = dc_low_raw.resample("30min").ffill().to_frame("value")
    dc_high_30 = dc_high_raw.resample("30min").ffill().to_frame("value")

    da_fc  = _load_forecast("da",      PERIOD)
    bm_fc  = _load_forecast("bm",      PERIOD)
    dcl_fc = _load_forecast("dc_low",  PERIOD)
    dch_fc = _load_forecast("dc_high", PERIOD)

    pool = build_error_path_pool(
        da_actual_30, bm_actual, dc_low_30, dc_high_30,
        da_fc, bm_fc, dcl_fc, dch_fc,
    )

    # ------------------------------------------------------------------
    # 1. The actual Dec 15 spread at the spike window — what perfect
    #    foresight is exploiting
    # ------------------------------------------------------------------
    target_start = pd.Timestamp(TARGET_DATE, tz="UTC")
    idx = pd.date_range(target_start, periods=48, freq="30min")

    da_actual_spike = da_actual_30["value"].reindex(idx).to_numpy()[SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]
    bm_actual_spike = bm_actual["value"].reindex(idx).to_numpy()[SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]
    actual_spread = bm_actual_spike - da_actual_spike

    print(f"\n{'='*65}")
    print(f"  ACTUAL Dec 15, 16:00-19:30 -- BM minus DA spread per period")
    print(f"{'='*65}")
    print(f"  {np.round(actual_spread, 1)}")
    print(f"  mean spread: {actual_spread.mean():.1f}   max spread: {actual_spread.max():.1f}")

    # ------------------------------------------------------------------
    # 2. The Dec 15 q50 forecast's implied spread at the same window --
    #    what the optimizer would see as "expected" before any error
    #    path is added
    # ------------------------------------------------------------------
    da_fc_win = da_fc.loc[idx[0]:idx[-1]]["q50"].to_numpy()[SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]
    bm_fc_win = bm_fc.loc[idx[0]:idx[-1]]["q50"].to_numpy()[SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]
    q50_spread = bm_fc_win - da_fc_win

    print(f"\n{'='*65}")
    print(f"  Dec 15 Q50 FORECAST, same window -- BM minus DA spread")
    print(f"{'='*65}")
    print(f"  {np.round(q50_spread, 1)}")
    print(f"  mean spread: {q50_spread.mean():.1f}   max spread: {q50_spread.max():.1f}")

    # ------------------------------------------------------------------
    # 3. The pool's error-path distribution at the SAME period positions
    #    (32-39) across all 1034 historical days -- this is what gets
    #    added to q50 to build a scenario. If the pool's typical/extreme
    #    error-implied spread at this window is much smaller than what
    #    would be needed to beat round-trip + degradation costs, that
    #    explains passivity regardless of which days get sampled.
    # ------------------------------------------------------------------
    da_err_window = pool["da"][:, SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]
    bm_err_window = pool["bm"][:, SPIKE_PERIOD_START:SPIKE_PERIOD_END+1]

    # error-implied spread change: how much would adding this day's errors
    # shift the BM-DA spread, relative to q50's own (already mean-zero-ish)
    # spread assumption
    err_spread_shift = bm_err_window - da_err_window  # shape (n_pool, 8)
    mean_shift_per_day = err_spread_shift.mean(axis=1)  # average shift across the window, per day

    print(f"\n{'='*65}")
    print(f"  POOL: error-implied spread SHIFT at the same window,")
    print(f"  across all {len(pool['dates'])} historical days")
    print(f"{'='*65}")
    print(f"  mean of mean_shift_per_day:   {mean_shift_per_day.mean():.2f}")
    print(f"  std  of mean_shift_per_day:   {mean_shift_per_day.std():.2f}")
    print(f"  min (most DA-favoring):       {mean_shift_per_day.min():.2f}")
    print(f"  max (most BM-favoring):       {mean_shift_per_day.max():.2f}")
    print(f"  95th percentile:              {np.percentile(mean_shift_per_day, 95):.2f}")
    print(f"  99th percentile:              {np.percentile(mean_shift_per_day, 99):.2f}")

    print(f"\n{'='*65}")
    print(f"  COMPARISON")
    print(f"{'='*65}")
    print(f"  Actual realised mean spread on Dec 15:        {actual_spread.mean():8.1f}")
    print(f"  Q50 forecast's own mean spread assumption:    {q50_spread.mean():8.1f}")
    print(f"  Pool's most extreme (99th pct) error shift:   {np.percentile(mean_shift_per_day, 99):8.1f}")
    print(f"  Pool's typical (mean) error shift:            {mean_shift_per_day.mean():8.1f}")
    print()
    implied_best_case = q50_spread.mean() + np.percentile(mean_shift_per_day, 99)
    print(f"  Best case achievable by sampling (q50 + 99th pct shift): {implied_best_case:8.1f}")
    print(f"  vs. what actually happened:                              {actual_spread.mean():8.1f}")
    print(
        "\n  If 'best case achievable by sampling' is well below 'what actually "
        "happened', that confirms the pool structurally cannot produce a "
        "scenario aggressive enough to justify full-power cycling on this "
        "specific day -- explaining seed-independent passivity directly."
    )
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()