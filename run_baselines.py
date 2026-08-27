"""
run_baselines.py — Compare the main model against three baseline strategies.

Strategies
----------
  main      — Gaussian copula with 20 scenarios (the full model)
  median    — Deterministic single scenario using q50 forecast values
  prev_day  — Deterministic single scenario using realised prices from 24 hours earlier
  prev_week — Deterministic single scenario using realised prices from 1 week earlier

All strategies use the same optimiser, constraints, battery spec, degradation
model, and first-24-hour settlement. Only the scenario assumption differs, so
any revenue difference is attributable to forecast quality and uncertainty
quantification alone.

Outputs (saved to data/results/)
---------------------------------
  backtest_main.parquet        — raw day-by-day results for each strategy
  backtest_median.parquet
  backtest_prev_day.parquet
  backtest_prev_week.parquet
  baseline_comparison.parquet  — aggregated metrics (all dates + clean dates)
  baseline_comparison.csv      — same, in CSV for easy inspection

Usage
-----
    uv run run_baselines.py --period oos        # 2025 OOS evaluation (primary)
    uv run run_baselines.py --period cv         # 2022-2024 CV evaluation
    uv run run_baselines.py --period oos --start 2025-01-01 --end 2025-06-30
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest
from baselines.scenarios import build_median_scenario, make_naive_scenario_builder
from forecasting.base import BaseForecaster
from run_backtest import _ensure_utc, _load_actual, _load_dc_actual, _load_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR    = Path("data")
RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _annualise(total_rev: float, n_days: int) -> float:
    return total_rev / n_days * 365 if n_days > 0 else float("nan")


def _compute_metrics(df: pd.DataFrame, label: str, clean_dates: set) -> dict:
    """Compute all comparison metrics for one strategy's results DataFrame."""
    solved = df[~df["solve_failed"].fillna(True)]

    # All evaluated (non-failed) days
    n_all         = len(solved)
    total_net_all = solved["net_revenue"].sum(skipna=True)

    # Clean days only — shared across all strategies
    clean_mask    = solved.index.isin(clean_dates)
    solved_clean  = solved[clean_mask]
    n_clean       = len(solved_clean)
    total_net_clean = solved_clean["net_revenue"].sum(skipna=True)

    return {
        "strategy":                     label,
        "n_evaluated_days":             n_all,
        "n_clean_days":                 n_clean,
        "total_net_revenue":            round(total_net_all, 2),
        "annualised_net_revenue_all":   round(_annualise(total_net_all,   n_all),   2),
        "annualised_net_revenue_clean": round(_annualise(total_net_clean, n_clean), 2),
        "avg_net_revenue_per_day":      round(solved["net_revenue"].mean(skipna=True), 2),
        "total_da_revenue":             round(solved["da_revenue"].sum(skipna=True),       2),
        "total_bm_revenue":             round(solved["bm_revenue"].sum(skipna=True),       2),
        "total_dc_low_revenue":         round(solved["dc_low_revenue"].sum(skipna=True),   2),
        "total_dc_high_revenue":        round(solved["dc_high_revenue"].sum(skipna=True),  2),
        "total_degradation_cost":       round(solved["degradation_cost"].sum(skipna=True), 2),
        "n_solve_failures":             int(df["solve_failed"].fillna(True).sum()),
    }


def _shared_clean_dates(*result_dfs: pd.DataFrame) -> set:
    """
    Return the set of dates where every strategy solved without failure and no
    DA imputation flag is set.

    All strategies are evaluated on exactly this set when computing the
    'clean dates' metrics — never each strategy's own clean subset, ensuring the metric is
    the same across strategies.
    """
    solved_sets = []
    for df in result_dfs:
        solved = df[~df["solve_failed"].fillna(True)]
        if "da_imputed" in solved.columns:
            solved = solved[~solved["da_imputed"].fillna(False)]
        solved_sets.append(set(solved.index))

    common = solved_sets[0]
    for s in solved_sets[1:]:
        common &= s

    logger.info(
        "Shared clean-date set: %d dates (all strategies solved + no DA imputation).",
        len(common),
    )
    return common


def _compute_held_out_forecast_metrics(da_actual, bm_actual, dc_low_actual, dc_high_actual,
                                       da_fc, bm_fc, dc_low_fc, dc_high_fc, clean_dates: set) -> pd.DataFrame:
    """Compute forecast evaluation metrics specifically on the shared clean-date set."""
    metrics_rows = []
    
    # Map markets to their actual and forecast frames
    markets = [
        ("DA (£/MWh)",      da_actual["value"],      da_fc),
        ("BM (£/MWh)",      bm_actual["value"],      bm_fc),
        ("DC Low (£/MW/h)",  dc_low_actual["value"],  dc_low_fc),
        ("DC High (£/MW/h)", dc_high_actual["value"], dc_high_fc),
    ]
    
    for name, actual_s, fc_df in markets:
        # Filter to clean dates
        clean_mask = fc_df.index.map(lambda ts: ts.date()).isin(clean_dates)
        fc_clean = fc_df[clean_mask]
        
        # Align actuals
        common_idx = fc_clean.index.intersection(actual_s.index)
        if len(common_idx) == 0:
            continue
            
        y_true = actual_s.loc[common_idx].to_numpy(dtype=float)
        q10 = fc_clean.loc[common_idx, "q10"].to_numpy(dtype=float)
        q50 = fc_clean.loc[common_idx, "q50"].to_numpy(dtype=float)
        q90 = fc_clean.loc[common_idx, "q90"].to_numpy(dtype=float)
        
        m = BaseForecaster.evaluate_forecast_metrics(y_true, q10, q50, q90)
        m["market"] = name
        m["n_periods"] = len(common_idx)
        metrics_rows.append(m)
        
    return pd.DataFrame(metrics_rows).set_index("market")

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_all(period: str, start_date: str, end_date: str) -> None:

    # ----------------------------------------------------------------
    # 1. Load actuals
    # ----------------------------------------------------------------
    logger.info("Loading actuals...")
    da_actual      = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual      = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")

    # DC actuals are at EFA-block resolution; run_backtest normalises them to
    # 30-min internally, but the naive scenario builders need 30-min versions
    # at factory construction time (the closure holds a reference to the frame
    # passed here, not to the engine's internal normalised copy).
    dc_low_raw  = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_actual  = dc_low_raw.to_frame("value")   # block-level for run_backtest
    dc_high_actual = dc_high_raw.to_frame("value")
    dc_low_30  = dc_low_raw.resample("30min").ffill().to_frame("value")   # 30-min for naive builders
    dc_high_30 = dc_high_raw.resample("30min").ffill().to_frame("value")

    # ----------------------------------------------------------------
    # 2. Load forecasts
    # ----------------------------------------------------------------
    logger.info("Loading %s forecasts...", period.upper())
    da_fc      = _load_forecast("da",      period)
    bm_fc      = _load_forecast("bm",      period)
    dc_low_fc  = _load_forecast("dc_low",  period)
    dc_high_fc = _load_forecast("dc_high", period)

    # ----------------------------------------------------------------
    # 3. Define strategies
    # ----------------------------------------------------------------
    # All builders share the same callable signature:
    #   fn(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win, seed)
    # scenario_builder=None means the main copula model is used.
    strategies = [
        ("main",      None),
        ("median",    build_median_scenario),
        ("prev_day",  make_naive_scenario_builder(
                          da_actual, bm_actual, dc_low_30, dc_high_30, lag_periods=48)),
        ("prev_week", make_naive_scenario_builder(
                          da_actual, bm_actual, dc_low_30, dc_high_30, lag_periods=336)),
    ]

    # Keyword arguments shared by all run_backtest calls
    shared_kwargs = {
        "da_actual": da_actual,
        "bm_actual": bm_actual,
        "dc_low_actual": dc_low_actual,
        "dc_high_actual": dc_high_actual,
        "da_forecast": da_fc,
        "bm_forecast": bm_fc,
        "dc_low_forecast": dc_low_fc,
        "dc_high_forecast": dc_high_fc,
        "start_date": start_date,
        "end_date": end_date,
        "seed": 42,
    }

    # ----------------------------------------------------------------
    # 4. Run each strategy; save raw results immediately
    # ----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, pd.DataFrame] = {}

    for label, builder in strategies:
        logger.info("Running strategy: %s ...", label)
        df = run_backtest(**shared_kwargs, scenario_builder=builder)

        if df.empty:
            logger.warning("No results for '%s' — skipping.", label)
            continue

        out_path = RESULTS_DIR / f"backtest_{label}_{period}.parquet"
        df.to_parquet(out_path)
        logger.info("  %d rows -> %s", len(df), out_path)

        failure_rate = df["solve_failed"].fillna(True).mean()
        if failure_rate > 0.05:
            logger.warning(
                "  Strategy '%s' has %.1f%% solve failures — results may be unreliable.",
                label, failure_rate * 100,
            )

        all_results[label] = df

    if len(all_results) < 2:
        logger.error(
            "Fewer than 2 strategies produced results — cannot build comparison table."
        )
        return

    # ----------------------------------------------------------------
    # 5. Compute shared clean-date set; assert all strategies cover it
    # ----------------------------------------------------------------
    clean_dates = _shared_clean_dates(*all_results.values())

    for label, df in all_results.items():
        n_clean_in_df = df.index.isin(clean_dates).sum()
        if n_clean_in_df != len(clean_dates):
            logger.warning(
                "Strategy '%s' only covers %d / %d clean dates — "
                "check date range alignment.",
                label, n_clean_in_df, len(clean_dates),
            )

    # ----------------------------------------------------------------
    # 6. Build comparison table
    # ----------------------------------------------------------------
    rows = [_compute_metrics(df, label, clean_dates) for label, df in all_results.items()]
    comparison = pd.DataFrame(rows).set_index("strategy")

    parquet_path = RESULTS_DIR / f"baseline_comparison_{period}.parquet"
    csv_path     = RESULTS_DIR / f"baseline_comparison_{period}.csv"
    comparison.to_parquet(parquet_path)
    comparison.to_csv(csv_path)
    logger.info("Comparison table -> %s and %s", parquet_path, csv_path)

    # Compute forecast metrics on the same clean dates
    fc_metrics = _compute_held_out_forecast_metrics(
        da_actual, bm_actual, dc_low_30, dc_high_30,
        da_fc, bm_fc, dc_low_fc, dc_high_fc, clean_dates
    )
    fc_csv_path = RESULTS_DIR / f"held_out_forecast_metrics_{period}.csv"
    fc_metrics.to_csv(fc_csv_path)
    logger.info("Held-out forecast metrics -> %s", fc_csv_path)

    # ----------------------------------------------------------------
    # 7. Print summary
    # ----------------------------------------------------------------
    display_cols = [
        "n_evaluated_days", "n_clean_days",
        "annualised_net_revenue_all", "annualised_net_revenue_clean",
        "avg_net_revenue_per_day", "n_solve_failures",
    ]

    print(f"\n{'='*72}")
    print(f"  Baseline comparison — {period.upper()}  ({start_date} -> {end_date})")
    print(f"  Shared clean-date set: {len(clean_dates)} days")
    print(f"{'='*72}")
    print(f"\n  Held-out Forecast Performance (on the SAME {len(clean_dates)} clean dates):")
    print(fc_metrics.to_string())
    print()
    print(comparison[display_cols].to_string())
    print()
    print("  Market revenue breakdown (£):")
    print(comparison[[
        "total_da_revenue", "total_bm_revenue",
        "total_dc_low_revenue", "total_dc_high_revenue",
        "total_degradation_cost", "total_net_revenue",
    ]].to_string())
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="BESS strategy baseline comparison")
    parser.add_argument(
        "--period", choices=["cv", "oos"], required=True,
        help="Backtest period: 'cv' (2022-2024 OOF) or 'oos' (2025 final)",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional override)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (optional override)")
    args = parser.parse_args()

    if args.period == "cv":
        start = args.start or "2022-01-01"
        end   = args.end   or "2024-12-31"
    else:
        start = args.start or "2025-01-01"
        end   = args.end   or "2025-12-31"

    run_all(period=args.period, start_date=start, end_date=end)


if __name__ == "__main__":
    main()
