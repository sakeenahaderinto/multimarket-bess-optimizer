"""
Run a walk-forward backtest using historical prices and forecasts.

Two backtest modes:
  - CV (2022-2024): Uses out-of-fold predictions from walk-forward CV (bias-free)
  - OOS (2025): Uses true out-of-sample forecasts from final models

Usage:
    uv run run_backtest.py --period cv    # 2022-2024 with OOF predictions
    uv run run_backtest.py --period oos   # 2025 with final model forecasts
    uv run run_backtest.py --period cv --start 2023-01-01 --end 2023-12-31
    uv run run_backtest.py --period oos --step 7   # weekly steps for 2025
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.engine import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def _ensure_utc(index: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    """Coerce a DatetimeIndex to UTC, localising if naive or converting if another tz."""
    if index.tz is None:
        logger.debug("Localising '%s' index to UTC (was timezone-naive).", name)
        return index.tz_localize("UTC")
    if str(index.tz) != "UTC":
        logger.warning(
            "Converting '%s' index from %s to UTC.", name, index.tz
        )
        return index.tz_convert("UTC")
    return index


def _load_actual(glob_pattern: str, value_col: str) -> pd.Series:
    """
    Load and concatenate parquet files matching glob_pattern.

    The datetime column is parsed with utc=True so the index is always
    UTC-aware, regardless of how the files were written.
    """
    files = sorted(DATA_DIR.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {DATA_DIR / glob_pattern}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # Support both 'datetime' and 'startTime' column names
        time_col = "datetime" if "datetime" in df.columns else "startTime"
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()
        if value_col not in df.columns:
            raise KeyError(
                f"Expected column '{value_col}' not found in {f}. "
                f"Available columns: {list(df.columns)}"
            )
        frames.append(df[[value_col]])

    combined = pd.concat(frames)
    # Drop duplicate timestamps (e.g. from overlapping parquet files)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index = _ensure_utc(combined.index, glob_pattern)
    return combined[value_col]


def _load_forecast(model_id: str, period: str) -> pd.DataFrame:
    """
    Load forecast file based on backtest period.

    The parquet index is coerced to UTC so it aligns with actuals loaded
    via _load_actual, which always produces a UTC-aware index.

    Args:
        model_id: Model identifier (da, bm, dc_low, dc_high)
        period: 'cv' for 2022-2024 OOF predictions, 'oos' for 2025 final forecasts
    """
    if period == "cv":
        path = DATA_DIR / "forecasts" / f"{model_id}_oof_2022_2024.parquet"
        period_desc = "CV out-of-fold (2022-2024)"
    elif period == "oos":
        path = DATA_DIR / "forecasts" / f"{model_id}_oos_2025.parquet"
        period_desc = "OOS final model (2025)"
    else:
        raise ValueError(f"Invalid period '{period}'. Must be 'cv' or 'oos'.")

    if not path.exists():
        raise FileNotFoundError(
            f"Forecast not found: {path}\n"
            f"Run 'uv run train_forecasters.py' first to generate {period_desc} forecasts."
        )

    df = pd.read_parquet(path)
    df.index = _ensure_utc(df.index, f"{model_id} forecast")
    # Drop duplicates in forecast index too
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _load_dc_actual(service: str) -> pd.Series:
    """
    Load DC auction results for a given service (DCL or DCH).

    Coerces delivery_start to UTC and forward-fills to 30-min resolution
    so the index aligns with DA/BM actuals before entering the engine.
    """
    path = DATA_DIR / "processed" / "dc_auction_processed.parquet"
    df = pd.read_parquet(path)
    df = df[df["service"] == service].copy()
    df["delivery_start"] = pd.to_datetime(df["delivery_start"], utc=True)
    s = (
        df.set_index("delivery_start")
        .sort_index()["clearing_price"]
    )
    s.index = _ensure_utc(s.index, f"dc_{service.lower()}_actual")
    return s


def _log_data_summary(name: str, s: pd.Series | pd.DataFrame) -> None:
    """Log a one-line summary of a loaded series/frame for quick sanity-checking."""
    idx = s.index
    logger.info(
        "  %-20s  rows=%-6d  %s → %s  tz=%s",
        name,
        len(s),
        idx.min().date() if len(s) else "N/A",
        idx.max().date() if len(s) else "N/A",
        idx.tz,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward BESS backtest")
    parser.add_argument(
        "--period",
        choices=["cv", "oos"],
        required=True,
        help="Backtest period: 'cv' (2022-2024 OOF) or 'oos' (2025 final)"
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional override)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (optional override)")
    parser.add_argument("--step",  type=int, default=1, help="Days per backtest step (default: 1)")
    parser.add_argument("--use-spread-model", action="store_true",
                        help="Use direct BM-DA spread model for scenario construction (BM = DA + spread)")
    args = parser.parse_args()

    # Enforce period boundaries if not explicitly overridden
    if args.period == "cv":
        default_start = "2022-01-01"
        default_end = "2024-12-31"
        if args.start and args.start >= "2025-01-01":
            logger.error("CV period is 2022-2024. Cannot use start date >= 2025-01-01.")
            return
        if args.end and args.end >= "2025-01-01":
            logger.warning("CV period ends 2024-12-31. Capping end date.")
            args.end = "2024-12-31"
    else:  # oos
        default_start = "2025-01-01"
        default_end = "2025-12-31"
        if args.start and args.start < "2025-01-01":
            logger.error("OOS period is 2025+. Cannot use start date < 2025-01-01.")
            return

    start_date = args.start or default_start
    end_date = args.end or default_end

    logger.info("Backtest period: %s (%s to %s)", args.period.upper(), start_date, end_date)

    logger.info("Loading actuals...")
    da_actual     = _load_actual("processed/day_ahead_processed.parquet", "value")
    bm_actual     = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice")
    dc_low_actual  = _load_dc_actual("DCL")
    dc_high_actual = _load_dc_actual("DCH")

    logger.info("Actual data summary:")
    _log_data_summary("da_actual",      da_actual)
    _log_data_summary("bm_actual",      bm_actual)
    _log_data_summary("dc_low_actual",  dc_low_actual)
    _log_data_summary("dc_high_actual", dc_high_actual)

    logger.info("Loading %s forecasts...", args.period.upper())
    da_fc      = _load_forecast("da",      args.period)
    bm_fc      = _load_forecast("bm",      args.period)
    dc_low_fc  = _load_forecast("dc_low",  args.period)
    dc_high_fc = _load_forecast("dc_high", args.period)

    spread_fc = None
    if args.use_spread_model:
        logger.info("Loading spread forecast...")
        spread_fc = _load_forecast("spread", args.period)
        _log_data_summary("spread_fc", spread_fc)

    logger.info("Forecast data summary:")
    _log_data_summary("da_fc",      da_fc)
    _log_data_summary("bm_fc",      bm_fc)
    _log_data_summary("dc_low_fc",  dc_low_fc)
    _log_data_summary("dc_high_fc", dc_high_fc)

    logger.info("Running backtest...")
    results = run_backtest(
        da_actual=da_actual.to_frame("value"),
        bm_actual=bm_actual.to_frame("value"),
        dc_low_actual=dc_low_actual.to_frame("value"),
        dc_high_actual=dc_high_actual.to_frame("value"),
        da_forecast=da_fc,
        bm_forecast=bm_fc,
        dc_low_forecast=dc_low_fc,
        dc_high_forecast=dc_high_fc,
        step_days=args.step,
        start_date=start_date,
        end_date=end_date,
        spread_forecast=spread_fc,
    )

    if results.empty:
        logger.warning("No results — check date range and data availability.")
        return

    output_file = f"backtest_{args.period}_{start_date}_{end_date}.parquet"
    output_path = DATA_DIR / "backtest" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_path)

    # Use nansum/nanmean so failed rows (NaN revenue) don't corrupt totals
    total = results["net_revenue"].sum(skipna=True)
    days  = results["solve_failed"].eq(False).sum()  # count only solved steps
    ann   = total / days * 365 if days else 0.0

    print(f"\n{'='*50}")
    print(f"  Backtest: {len(results)} steps  ({results.index.min()} → {results.index.max()})")
    print(f"  {'DA revenue':30s}  £{results['da_revenue'].sum(skipna=True):>10,.0f}")
    print(f"  {'BM revenue':30s}  £{results['bm_revenue'].sum(skipna=True):>10,.0f}")
    print(f"  {'DC Low revenue':30s}  £{results['dc_low_revenue'].sum(skipna=True):>10,.0f}")
    print(f"  {'DC High revenue':30s}  £{results['dc_high_revenue'].sum(skipna=True):>10,.0f}")
    print(f"  {'Degradation cost':30s}  £{results['degradation_cost'].sum(skipna=True):>10,.0f}")
    print(f"  {'Net revenue':30s}  £{total:>10,.0f}")
    print(f"  {'Annualised (approx)':30s}  £{ann:>10,.0f}")
    print(f"  Solve failures: {results['solve_failed'].sum()} / {len(results)}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()