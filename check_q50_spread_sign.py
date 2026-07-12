"""
check_q50_spread_sign.py — Is the q50 BM-DA spread sign error specific to
2022-12-15, or does it happen systematically on other high-volatility days?

For every day in the test period, compute:
    actual_spread_mean = mean(bm_actual - da_actual) over the day
    q50_spread_mean    = mean(bm_q50    - da_q50)    over the day
and check whether q50 got the SIGN right (not just the magnitude).

Supports swapping in the new hourly-trained DA forecaster (upsampled back
to 30-min) in place of the original 30-min-trained DA forecaster, to test
whether training at native hourly resolution also improves the
cross-market spread-sign accuracy, or whether that problem is independent
of the duplication artifact.

Usage:
    uv run check_q50_spread_sign.py                  # original 30-min DA
    uv run check_q50_spread_sign.py --use-hourly-da   # new hourly DA, upsampled
"""

import argparse

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from run_backtest import _load_actual, _load_forecast
from features.pipeline_da_hourly import upsample_da_forecast
from optimiser.scenarios import _efa_block_groups

PERIOD = "cv"
DEC_2022_START = "2022-12-01"
DEC_2022_END = "2022-12-31"


def _daily_spread_table(da_actual, bm_actual, da_fc, bm_fc, dates, spread_fc=None) -> pd.DataFrame:
    rows = []
    for d in dates:
        day_start = pd.Timestamp(d, tz="UTC")
        idx = pd.date_range(day_start, periods=48, freq="30min")

        da_a = da_actual["value"].reindex(idx)
        bm_a = bm_actual["value"].reindex(idx)

        if da_a.isna().any() or bm_a.isna().any():
            continue

        if spread_fc is not None:
            sp = spread_fc.reindex(idx)["q50"]
            if sp.isna().any():
                continue
            q50_spread = sp.mean()
        else:
            da_q = da_fc.reindex(idx)["q50"]
            bm_q = bm_fc.reindex(idx)["q50"]
            if da_q.isna().any() or bm_q.isna().any():
                continue
            q50_spread = (bm_q - da_q).mean()

        actual_spread = (bm_a - da_a).mean()

        rows.append({
            "date": d,
            "actual_spread_mean": actual_spread,
            "q50_spread_mean": q50_spread,
            "sign_match": np.sign(actual_spread) == np.sign(q50_spread),
            "actual_volatility": (bm_a - da_a).std(),
        })

    return pd.DataFrame(rows).set_index("date")

def _compute_all_mwsa(
    da_actual_30: pd.DataFrame,
    bm_actual: pd.DataFrame,
    da_fc,
    bm_fc,
    dates,
    spread_fc=None,
) -> dict:
    """
    Compute magnitude-weighted sign accuracy at three granularities:
      period (30-min), EFA block (4h), and daily.
    """
    period_actual, period_q50 = [], []
    block_actual,  block_q50  = [], []
    daily_actual,  daily_q50  = [], []

    for d in dates:
        day_start = pd.Timestamp(d, tz="UTC")
        idx = pd.date_range(day_start, periods=48, freq="30min")

        da_a = da_actual_30["value"].reindex(idx)
        bm_a = bm_actual["value"].reindex(idx)
        if da_a.isna().any() or bm_a.isna().any():
            continue

        if spread_fc is not None:
            sp_q50 = spread_fc.reindex(idx)["q50"]
            if sp_q50.isna().any():
                continue
        else:
            da_q = da_fc.reindex(idx)["q50"]
            bm_q = bm_fc.reindex(idx)["q50"]
            if da_q.isna().any() or bm_q.isna().any():
                continue
            sp_q50 = bm_q - da_q

        actual_s = (bm_a - da_a).values   # (48,) actual spread per period
        q50_s    = sp_q50.values           # (48,) forecast spread per period

        # Period level
        period_actual.extend(actual_s)
        period_q50.extend(q50_s)

        # EFA block level — use real block boundaries, not midnight-aligned groups
        for blk_start, blk_end in _efa_block_groups(day_start, 48):
            block_actual.append(actual_s[blk_start:blk_end].mean())
            block_q50.append(q50_s[blk_start:blk_end].mean())

        # Daily level
        daily_actual.append(actual_s.mean())
        daily_q50.append(q50_s.mean())

    def _mwsa(actuals, forecasts):
        a = np.array(actuals)
        f = np.array(forecasts)
        w = np.abs(a)
        return float((w * (np.sign(a) == np.sign(f))).sum() / w.sum()) if w.sum() > 0 else float("nan")

    return {
        "period": _mwsa(period_actual, period_q50),
        "block":  _mwsa(block_actual,  block_q50),
        "daily":  _mwsa(daily_actual,  daily_q50),
        "n_periods": len(period_actual),
        "n_blocks":  len(block_actual),
        "n_days":    len(daily_actual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check q50 BM-DA spread sign accuracy")
    parser.add_argument(
        "--use-hourly-da", action="store_true",
        help="Use the hourly-trained DA forecaster (upsampled to 30-min) instead of the original 30-min DA model",
    )
    parser.add_argument(
        "--use-spread-model", action="store_true",
        help="Use the direct BM-DA spread model (spread_q50) instead of implied spread (bm_q50 - da_q50)",
    )
    args = parser.parse_args()

    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual")

    if args.use_spread_model:
        print("Using direct spread model (spread_q50).")
        spread_fc = _load_forecast("spread", PERIOD)
        da_fc = None
        bm_fc = None
        fc_index = spread_fc.index
    else:
        spread_fc = None
        if args.use_hourly_da:
            print("Using hourly-trained DA forecaster (da_hourly), upsampled to 30-min.")
            da_fc_hourly = _load_forecast("da_hourly", PERIOD)
            da_fc = upsample_da_forecast(da_fc_hourly)
        else:
            print("Using original 30-min-trained DA forecaster (da).")
            da_fc = _load_forecast("da", PERIOD)
        bm_fc = _load_forecast("bm", PERIOD)
        fc_index = da_fc.index

    # ------------------------------------------------------------------
    # December 2022 specifically
    # ------------------------------------------------------------------
    dec_dates = pd.date_range(DEC_2022_START, DEC_2022_END, freq="D").date
    dec_table = _daily_spread_table(da_actual_30, bm_actual, da_fc, bm_fc, dec_dates, spread_fc=spread_fc)

    print(f"\n{'='*70}")
    print(f"  DECEMBER 2022 -- daily BM-DA spread, actual vs q50 sign match")
    print(f"{'='*70}")
    print(dec_table.round(1).to_string())

    n_mismatch = (~dec_table["sign_match"]).sum()
    print(f"\n  Sign mismatches: {n_mismatch} / {len(dec_table)} days "
          f"({n_mismatch / len(dec_table) * 100:.1f}%)")
    weights_dec = dec_table["actual_spread_mean"].abs()
    mw_acc_dec = (weights_dec * dec_table["sign_match"]).sum() / weights_dec.sum()
    print(f"  Magnitude-weighted sign accuracy: {mw_acc_dec * 100:.1f}%  "
          f"(raw: {(1 - n_mismatch / len(dec_table)) * 100:.1f}%)")
    # mwsa -> magnitude-weighted sign accuracy
    dec_mwsa = _compute_all_mwsa(da_actual_30, bm_actual, da_fc, bm_fc, dec_dates, spread_fc=spread_fc)
    print(f"\n  Three-level MWSA — December 2022:")
    print(f"    Period-level (30-min) : {dec_mwsa['period']*100:.1f}%  ({dec_mwsa['n_periods']} periods)")
    print(f"    Block-level  (EFA 4h) : {dec_mwsa['block']*100:.1f}%  ({dec_mwsa['n_blocks']} blocks)")
    print(f"    Daily-level           : {dec_mwsa['daily']*100:.1f}%  ({dec_mwsa['n_days']} days)")


    # ------------------------------------------------------------------
    # Full CV period
    # ------------------------------------------------------------------
    full_dates = pd.date_range(fc_index.min().date(), fc_index.max().date(), freq="D").date
    full_table = _daily_spread_table(da_actual_30, bm_actual, da_fc, bm_fc, full_dates, spread_fc=spread_fc)

    n_mismatch_full = (~full_table["sign_match"]).sum()
    print(f"\n{'='*70}")
    print(f"  FULL CV PERIOD -- for context")
    print(f"{'='*70}")
    print(f"  Total usable days: {len(full_table)}")
    print(f"  Sign mismatches:   {n_mismatch_full} ({n_mismatch_full / len(full_table) * 100:.1f}%)")

    weights_full = full_table["actual_spread_mean"].abs()
    mw_acc_full = (weights_full * full_table["sign_match"]).sum() / weights_full.sum()
    print(f"  Magnitude-weighted sign accuracy: {mw_acc_full * 100:.1f}%  "
          f"(raw: {(1 - n_mismatch_full / len(full_table)) * 100:.1f}%)")

    # Split by volatility quartile
    full_table["vol_quartile"] = pd.qcut(full_table["actual_volatility"], 4, labels=["Q1 (calm)", "Q2", "Q3", "Q4 (volatile)"])
    by_quartile = full_table.groupby("vol_quartile")["sign_match"].agg(["mean", "count"])
    by_quartile["mismatch_rate_pct"] = (1 - by_quartile["mean"]) * 100

    def _mw_acc(g):
        w = g["actual_spread_mean"].abs()
        return (w * g["sign_match"]).sum() / w.sum() * 100

    by_quartile["mw_accuracy_pct"] = full_table.groupby("vol_quartile", observed=True).apply(_mw_acc)

    print(f"\n  Sign-mismatch rate by realised-volatility quartile:")
    print(by_quartile[["count", "mismatch_rate_pct", "mw_accuracy_pct"]].round(1).to_string())
    print(f"{'='*70}\n")

    full_mwsa = _compute_all_mwsa(da_actual_30, bm_actual, da_fc, bm_fc, full_dates, spread_fc=spread_fc)
    print(f"\n  Three-level MWSA — Full CV period:")
    print(f"    Period-level (30-min) : {full_mwsa['period']*100:.1f}%  ({full_mwsa['n_periods']} periods)")
    print(f"    Block-level  (EFA 4h) : {full_mwsa['block']*100:.1f}%  ({full_mwsa['n_blocks']} blocks)")
    print(f"    Daily-level           : {full_mwsa['daily']*100:.1f}%  ({full_mwsa['n_days']} days)")



if __name__ == "__main__":
    main()
