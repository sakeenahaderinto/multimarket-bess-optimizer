"""
calibrate_spread.py — Post-hoc interval calibration for the spread forecaster.

The spread model achieves 88.3% OOS coverage with nominal q10/q90 targets,
meaning the 80% predictive interval is too wide. This script:
  1. Finds the scaling factor s on OOF data such that
     [q50 - s*(q50-q10), q50 + s*(q90-q50)] achieves 80% empirical coverage.
  2. Overwrites the OOF and OOS parquet files with calibrated q10/q90 columns.
  3. Reports coverage and pinball loss before and after.

Run after train_spread.py. Re-running train_spread.py will overwrite calibration.

Usage:
    uv run calibrate_spread.py
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from backtest.engine import _normalise_to_30min
from run_backtest import _load_actual, _load_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
TARGET_COVERAGE = 0.80


def _pinball(actual, quantile_fc, q):
    err = actual - quantile_fc
    return np.where(err >= 0, q * err, (q - 1) * err).mean()


def find_scaling_factor(actual, q10, q50, q90, target=TARGET_COVERAGE):
    """Binary search for s such that [q50 ± s*half_width] gives target coverage."""
    lo, hi = 0.0, 2.0
    for _ in range(60):
        s = (lo + hi) / 2
        lower = q50 - s * (q50 - q10)
        upper = q50 + s * (q90 - q50)
        if ((actual >= lower) & (actual <= upper)).mean() > target:
            hi = s
        else:
            lo = s
    return (lo + hi) / 2


def apply_scaling(df: pd.DataFrame, s: float) -> pd.DataFrame:
    df = df.copy()
    df["q10"] = df["q50"] - s * (df["q50"] - df["q10"])
    df["q90"] = df["q50"] + s * (df["q90"] - df["q50"])
    return df

def compute_rolling_s(oof: pd.DataFrame, act: pd.Series, min_days: int = 90) -> dict:
    """
    For each date in oof, compute s using only errors strictly before that date.
    Returns a dict {date: s_value}. Defaults to s=1.0 for dates with < min_days history.
    """
    dates = sorted(set(oof.index.date))
    oof_dates = np.array(oof.index.date)
    s_by_date = {}
    for d in dates:
        mask = oof_dates < d
        n_past_days = mask.sum() // 48
        if n_past_days < min_days:
            s_by_date[d] = 1.0
        else:
            s_by_date[d] = find_scaling_factor(
                act.values[mask],
                oof["q10"].values[mask],
                oof["q50"].values[mask],
                oof["q90"].values[mask],
            )
    return s_by_date



def main():
    # Build actual spread series
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    da_30 = _normalise_to_30min(da_actual, "da_actual")
    spread_actual = (bm_actual["value"] - da_30["value"]).dropna()

    # Align OOF forecast with actuals
    oof = _load_forecast("spread", "cv")
    common = oof.index.intersection(spread_actual.index)
    oof = oof.reindex(common).dropna()
    act = spread_actual.reindex(oof.index).dropna()
    oof = oof.reindex(act.index)

    q10, q50, q90 = oof["q10"].values, oof["q50"].values, oof["q90"].values
    a = act.values

    # Metrics before
    below_q10_raw = (a < q10).mean()
    above_q90_raw = (a > q90).mean()
    cov_before     = 1.0 - below_q10_raw - above_q90_raw
    logger.info("---- Before calibration ----")
    logger.info("  Lower-tail miss rate : %.1f%%  (target 10.0%%)", below_q10_raw * 100)
    logger.info("  Upper-tail miss rate : %.1f%%  (target 10.0%%)", above_q90_raw * 100)
    logger.info("  Total coverage       : %.1f%%  (target 80.0%%)", cov_before * 100)
    logger.info("  Pinball q10 : %.4f", _pinball(a, q10, 0.10))
    logger.info("  Pinball q90 : %.4f", _pinball(a, q90, 0.90))



    # Rolling calibration: compute s per date using only past data
    oof_dates = np.array(oof.index.date)
    s_by_date = compute_rolling_s(oof, act)

    # Apply date-specific s to produce calibrated quantiles
    oof_cal = oof.copy()
    for d, s_val in s_by_date.items():
        mask = oof_dates == d
        oof_cal.loc[mask, "q10"] = oof["q50"].values[mask] - s_val * (oof["q50"].values[mask] - oof["q10"].values[mask])
        oof_cal.loc[mask, "q90"] = oof["q50"].values[mask] + s_val * (oof["q90"].values[mask] - oof["q50"].values[mask])

    q10_cal = oof_cal["q10"].values
    q90_cal = oof_cal["q90"].values

    s_values = np.array(list(s_by_date.values()))
    logger.info("Rolling s: min=%.4f, median=%.4f, max=%.4f (first 90 days s=1.0 by design)",
                s_values.min(), np.median(s_values), s_values.max())

    below_q10 = (a < q10_cal).mean()
    above_q90 = (a > q90_cal).mean()
    cov_after  = 1.0 - below_q10 - above_q90
    logger.info("---- After rolling calibration ----")
    logger.info("  Lower-tail miss rate : %.1f%%  (target 10.0%%)", below_q10 * 100)
    logger.info("  Upper-tail miss rate : %.1f%%  (target 10.0%%)", above_q90 * 100)
    logger.info("  Total coverage       : %.1f%%  (target 80.0%%)", cov_after * 100)
    logger.info("  Pinball q10 : %.4f", _pinball(a, q10_cal, 0.10))
    logger.info("  Pinball q90 : %.4f", _pinball(a, q90_cal, 0.90))

    # Save calibrated OOF
    oof_cal.to_parquet(DATA_DIR / "forecasts" / "spread_calibrated_oof_2022_2024.parquet")
    logger.info("Calibrated OOF -> spread_calibrated_oof_2022_2024.parquet  [rolling s, no leakage]")

    # OOS: use global s from full OOF (appropriate — OOS is truly held-out)
    s_global = find_scaling_factor(a, q10, q50, q90)
    logger.info("Global s for OOS application: %.4f", s_global)
    oos_fc = _load_forecast("spread", "oos")
    oos_cal = apply_scaling(oos_fc, s_global)
    oos_cal.to_parquet(DATA_DIR / "forecasts" / "spread_calibrated_oos_2025.parquet")
    logger.info("Calibrated OOS -> spread_calibrated_oos_2025.parquet  [global s from full OOF is valid here]")




if __name__ == "__main__":
    main()
