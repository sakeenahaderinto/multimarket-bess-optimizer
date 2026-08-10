"""
run_fixed_month_comparison.py — Phase 1 fixed-month scenario comparison.

Runs all four planned cases on December 2022 (the chosen fixed test month:
highest DA volatility of any fully-clean month in the CV period, std=152.07,
includes a negative-price excursion down to -£44.51 and a spike up to
£1860.85):

  - perfect_foresight        — theoretical ceiling, realized prices, no uncertainty
  - median                   — deterministic q50-only (existing strategy)
  - current_scenarios        — existing Gaussian copula, independent per-period draws
  - historical_error_scenarios — samples joint historical forecast-error
                                  days and adds them to q50, preserving real
                                  temporal shape and cross-market co-movement
                                  without relying on an estimated correlation matrix

Usage:
    uv run run_fixed_month_comparison.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, _normalise_to_30min, DEFAULT_OPT_SETTINGS
from baselines.scenarios import build_median_scenario, _col, _block_avg_1d
from run_backtest import _load_actual, _load_dc_actual, _load_forecast
from historical_error_scenarios import build_error_path_pool, make_historical_error_scenario_builder
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Perfect foresight scenario builder
# ---------------------------------------------------------------------------


def make_perfect_foresight_builder(da_actual, bm_actual, dc_low_actual_30, dc_high_actual_30):
    """
    Returns a scenario_builder using realized actual prices instead of any
    forecast — the single scenario *is* what truly happened, so the optimizer
    faces no uncertainty. Establishes the revenue ceiling for the fixed-month
    comparison.

    Matches the n=1, shape-(1, horizon) convention used by build_median_scenario
    and the naive lag builders elsewhere in baselines/scenarios.py — n=1 is
    mathematically equivalent to n identical tiled copies under the optimizer's
    mean-across-scenarios objective, and avoids the unnecessary extra solve cost.

    DC Low/High are block-averaged via _block_avg_1d for strict EFA-block
    constancy in the scenario, matching the treatment in build_median_scenario
    and make_naive_scenario_builder.
    """
    def builder(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win, seed=None):
        if horizon % 8 != 0:
            raise ValueError(
                f"horizon={horizon} must be divisible by 8 for DC block averaging."
            )

        idx = pd.date_range(window_start, periods=horizon, freq="30min", tz=window_start.tzinfo)

        da_path      = _col(da_actual).reindex(idx)
        bm_path      = _col(bm_actual).reindex(idx)
        dc_low_path  = _col(dc_low_actual_30).reindex(idx)
        dc_high_path = _col(dc_high_actual_30).reindex(idx)

        missing = {
            "da": int(da_path.isna().sum()),
            "bm": int(bm_path.isna().sum()),
            "dc_low": int(dc_low_path.isna().sum()),
            "dc_high": int(dc_high_path.isna().sum()),
        }
        if any(missing.values()):
            raise ValueError(
                f"Perfect foresight requires complete actuals for the window "
                f"{window_start} to {window_start + pd.Timedelta(minutes=30*horizon)}. "
                f"Missing values: {missing}."
            )

        dcl_vals = _block_avg_1d(dc_low_path.to_numpy(dtype=float), horizon, window_start=window_start)
        dch_vals = _block_avg_1d(dc_high_path.to_numpy(dtype=float), horizon, window_start=window_start)

        return {
            "da":      da_path.to_numpy(dtype=float)[np.newaxis, :],
            "bm":      bm_path.to_numpy(dtype=float)[np.newaxis, :],
            "dc_low":  dcl_vals[np.newaxis, :],
            "dc_high": dch_vals[np.newaxis, :],
            "seed":    seed,
        }
    return builder


def make_q50_anchored_builder(base_builder, anchor_fraction: float):
    """
    Wraps any scenario builder and replaces the first `anchor_fraction` of scenarios
    with the pure q50 (zero-error) path.
    Tests the gradient from fully stochastic to fully deterministic (median).
    anchor_fraction is applied to the actual scenario count at build time so
    experiments remain comparable across different --n-scenarios values.
    """
    from optimiser.scenarios import _efa_block_groups

    def builder(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win, seed=None):
        scenarios = base_builder(
            window_start, horizon, da_fc_win, bm_fc_win,
            dc_low_fc_win, dc_high_fc_win, seed=seed,
        )

        efa_groups = _efa_block_groups(window_start, horizon)
        def _blk(arr):
            result = np.empty_like(arr, dtype=float)
            for s, e in efa_groups:
                result[s:e] = arr[s:e].mean()
            return result

        da_q50  = da_fc_win["q50"].values[:horizon]
        bm_q50  = bm_fc_win["q50"].values[:horizon]
        dcl_q50 = _blk(dc_low_fc_win["q50"].values[:horizon])
        dch_q50 = _blk(dc_high_fc_win["q50"].values[:horizon])

        k = max(1, round(anchor_fraction * scenarios["da"].shape[0]))
        scenarios["da"][:k]      = da_q50
        scenarios["bm"][:k]      = bm_q50
        scenarios["dc_low"][:k]  = dcl_q50
        scenarios["dc_high"][:k] = dch_q50

        return scenarios

    builder.__name__ = f"q50_anchored_f{round(anchor_fraction * 100):02d}"
    return builder


# ---------------------------------------------------------------------------
# Metrics helper (same shape as run_baselines.py's _compute_metrics)
# ---------------------------------------------------------------------------


def _compute_metrics(df: pd.DataFrame, label: str) -> dict:
    solved = df[~df["solve_failed"].fillna(True)]
    n_days = len(solved)
    total_net = solved["net_revenue"].sum(skipna=True)

    net_revs = solved["net_revenue"].dropna()
    worst_10pct_n = max(1, int(len(net_revs) * 0.1))
    avg_worst_10pct = net_revs.nsmallest(worst_10pct_n).mean() if len(net_revs) else float("nan")

    return {
        "strategy":               label,
        "n_evaluated_days":       n_days,
        "total_net_revenue":      round(total_net, 2),
        "avg_net_revenue_per_day": round(net_revs.mean(), 2) if len(net_revs) else float("nan"),
        "worst_daily_revenue":    round(net_revs.min(), 2) if len(net_revs) else float("nan"),
        "p5_revenue":             round(net_revs.quantile(0.05), 2) if len(net_revs) else float("nan"),
        "avg_worst_10pct_revenue": round(avg_worst_10pct, 2),
        "total_da_revenue":       round(solved["da_revenue"].sum(skipna=True), 2),
        "total_bm_revenue":       round(solved["bm_revenue"].sum(skipna=True), 2),
        "total_dc_low_revenue":   round(solved["dc_low_revenue"].sum(skipna=True), 2),
        "total_dc_high_revenue":  round(solved["dc_high_revenue"].sum(skipna=True), 2),
        "total_degradation_cost": round(solved["degradation_cost"].sum(skipna=True), 2),
        "n_solve_failures":       int(df["solve_failed"].fillna(True).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fixed-month scenario comparison")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--label", required=True, help="Output folder label, e.g. 'may2024'")
    parser.add_argument("--period", choices=["cv", "oos"], default="cv",
                        help="'cv' for 2022-2024 OOF, 'oos' for 2025 final forecasts")
    parser.add_argument("--error-cap-pct", type=float, default=0.95,
                        help="Error magnitude cap percentile for historical error scenarios (0=off, default=0.95)")

    parser.add_argument("--use-spread-model", action="store_true",
                        help="Include spread_scenarios strategy (requires spread forecasts)")
    parser.add_argument("--calibrated-spread", action="store_true",
                        help="Load spread_calibrated_*.parquet instead of spread_*.parquet (requires --use-spread-model)")
    parser.add_argument("--n-scenarios", type=int, default=20,
                        help="Scenarios per day (recommend ≥50 for stable CVaR)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds. Multiple seeds -> results averaged.")

    args = parser.parse_args()

    if args.calibrated_spread and not args.use_spread_model:
        parser.error("--calibrated-spread requires --use-spread-model")

    START_DATE = args.start
    END_DATE = args.end
    PERIOD = args.period

    seeds_str = "-".join(str(s) for s in args.seeds)
    tags = [f"n{args.n_scenarios}", f"s{seeds_str}"]
    if args.use_spread_model:
        tags.append("spread_cal" if args.calibrated_spread else "spread")
    OUTPUT_DIR = DATA_DIR / "results" / f"phase1_{args.label}_{'_'.join(tags)}"

    logger.info("Phase 1 fixed-month comparison: %s to %s", START_DATE, END_DATE)


    # ------------------------------------------------------------------
    # 1. Load actuals
    # ------------------------------------------------------------------
    logger.info("Loading actuals...")
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")

    # da_actual is natively hourly. engine.py's run_backtest() normalises its
    # own internal copy to 30-min before settlement (the fix applied earlier
    # in this project), but make_perfect_foresight_builder reindexes directly
    # against whatever da_actual is passed to it here — so it needs its own
    # pre-normalised, gap-free 30-min copy to avoid hitting the same
    # hourly-vs-30-min misalignment that engine.py had before being fixed.
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual")

    dc_low_raw  = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_actual  = dc_low_raw.to_frame("value")
    dc_high_actual = dc_high_raw.to_frame("value")
    dc_low_30  = dc_low_raw.resample("30min").ffill().to_frame("value")
    dc_high_30 = dc_high_raw.resample("30min").ffill().to_frame("value")

    # ------------------------------------------------------------------
    # 2. Load forecasts (CV / OOF, since Dec 2022 falls in that period)
    # ------------------------------------------------------------------
    logger.info("Loading %s forecasts...", PERIOD.upper())
    da_fc      = _load_forecast("da",      PERIOD)
    bm_fc      = _load_forecast("bm",      PERIOD)
    dc_low_fc  = _load_forecast("dc_low",  PERIOD)
    dc_high_fc = _load_forecast("dc_high", PERIOD)
    if args.use_spread_model:
        spread_model_id = "spread_calibrated" if args.calibrated_spread else "spread"
        spread_fc = _load_forecast(spread_model_id, PERIOD)
    else:
        spread_fc = None

    # ------------------------------------------------------------------
    # 3. Build the historical error-path pool (Phase 2)
    #    Built once, up front — reused across every day in the month,
    #    same pattern as _estimate_correlation_matrix() being computed
    #    once for current_scenarios rather than per-window.
    # ------------------------------------------------------------------
    logger.info("Building historical error-path pool...")
    error_pool = build_error_path_pool(
        da_actual_30, bm_actual, dc_low_30, dc_high_30,
        da_fc, bm_fc, dc_low_fc, dc_high_fc,
    )

    # ------------------------------------------------------------------
    # 4. Define all four cases
    # ------------------------------------------------------------------
    _hist_builder = make_historical_error_scenario_builder(
        error_pool, n=args.n_scenarios, da_actual=da_actual_30, recency_halflife=90,
        error_cap_pct=args.error_cap_pct,
    )

    # Tuple: (label, scenario_builder, spread_forecast, extra_opt_settings, is_deterministic)
    # Deterministic strategies produce the same result for every seed — run once only.
    strategies = [
        ("perfect_foresight", make_perfect_foresight_builder(
            da_actual_30, bm_actual, dc_low_30, dc_high_30), None, None, True),
        ("median",                       build_median_scenario, None, None, True),
        ("current_scenarios",          None, None, None, False),
        ("historical_error_scenarios",   _hist_builder, None, None, False),
        ("q50_anchor_f05", make_q50_anchored_builder(_hist_builder, anchor_fraction=0.05), None, None, False),
        ("q50_anchor_f25", make_q50_anchored_builder(_hist_builder, anchor_fraction=0.25), None, None, False),
        ("q50_anchor_f50", make_q50_anchored_builder(_hist_builder, anchor_fraction=0.50), None, None, False),
        ("cvar_l07_a09",  _hist_builder, None, {"cvar_alpha": 0.9, "mean_weight": 0.7}, False),
        ("cvar_l05_a09",  _hist_builder, None, {"cvar_alpha": 0.9, "mean_weight": 0.5}, False),
        ("cvar_l05_a08",  _hist_builder, None, {"cvar_alpha": 0.8, "mean_weight": 0.5}, False),
    ]
    if args.use_spread_model:
        strategies.append(("spread_scenarios", None, spread_fc, None, False))



    shared_kwargs = dict(
        da_actual=da_actual,
        bm_actual=bm_actual,
        dc_low_actual=dc_low_actual,
        dc_high_actual=dc_high_actual,
        da_forecast=da_fc,
        bm_forecast=bm_fc,
        dc_low_forecast=dc_low_fc,
        dc_high_forecast=dc_high_fc,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    base_settings = DEFAULT_OPT_SETTINGS.copy()
    base_settings["n_scenarios"] = args.n_scenarios

    # ------------------------------------------------------------------
    # 4. Run each strategy
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    rev_cols = ["net_revenue", "da_revenue", "bm_revenue",
                "dc_low_revenue", "dc_high_revenue", "degradation_cost"]

    for label, builder, sf, extra_settings, is_deterministic in strategies:
        logger.info("Running strategy: %s ...", label)
        eff_settings = base_settings.copy()
        if extra_settings:
            eff_settings.update(extra_settings)

        # Deterministic strategies produce identical output for every seed — run once.
        seeds_to_run = [args.seeds[0]] if is_deterministic else args.seeds

        per_seed_dfs = []
        for seed in seeds_to_run:
            df = run_backtest(**shared_kwargs, seed=seed,
                            scenario_builder=builder, spread_forecast=sf,
                            opt_settings=eff_settings)
            if not df.empty:
                per_seed_dfs.append(df)

        if not per_seed_dfs:
            logger.warning("No results for '%s' — skipping.", label)
            continue

        if len(per_seed_dfs) == 1:
            result_df = per_seed_dfs[0]
        else:
            # Align by date index before averaging — positional alignment breaks
            # when seeds have different solve-failure patterns.
            common_idx = per_seed_dfs[0].index
            for d in per_seed_dfs[1:]:
                common_idx = common_idx.intersection(d.index)

            result_df = per_seed_dfs[0].reindex(common_idx).copy()
            for col in rev_cols:
                if col in result_df.columns:
                    vals = np.array(
                        [d.reindex(common_idx)[col].values for d in per_seed_dfs],
                        dtype=float,
                    )
                    result_df[col] = np.nanmean(vals, axis=0)

            # solve_failed: True on any date where any seed failed
            result_df["solve_failed"] = False
            for d in per_seed_dfs:
                result_df["solve_failed"] = (
                    result_df["solve_failed"]
                    | d.reindex(common_idx)["solve_failed"].fillna(True)
                )

        out_path = OUTPUT_DIR / f"backtest_{label}.parquet"
        result_df.to_parquet(out_path)
        logger.info("  %d rows -> %s", len(result_df), out_path)
        all_results[label] = result_df

    # ------------------------------------------------------------------
    # 5. Build and print comparison table
    # ------------------------------------------------------------------
    # Use a shared clean-date set: only dates solved by every strategy.
    # This prevents strategies with more failures from being evaluated on
    # a higher-quality subset of dates than their competitors.
    shared_clean = None
    for df in all_results.values():
        clean = set(df[~df["solve_failed"].fillna(True)].index)
        shared_clean = clean if shared_clean is None else (shared_clean & clean)
    shared_clean = shared_clean or set()
    logger.info("Shared clean dates across all strategies: %d", len(shared_clean))

    rows = [
        _compute_metrics(df[df.index.isin(shared_clean)], label)
        for label, df in all_results.items()
    ]
    comparison = pd.DataFrame(rows).set_index("strategy")

    comparison_path = OUTPUT_DIR / "phase1_comparison.csv"
    comparison.to_csv(comparison_path)

    print(f"\n{'='*72}")
    print(f"  Phase 1 fixed-month comparison — {START_DATE} to {END_DATE}")
    print(f"{'='*72}")
    print(comparison.to_string())
    print(f"\n  Saved to: {comparison_path}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()