"""
inspect_day_schedule.py — Pull the actual dispatch schedule for one backtest
day, for direct comparison against realised prices.

run_backtest() solves a fresh Pyomo model per day and discards it after
settle_revenue() extracts the daily totals — the per-period decision
variables (da_charge, da_discharge, bm_offer, dc_low, dc_high) are never
saved. This script mirrors run_backtest's single-day logic but keeps the
solved model around so we can read those variables directly.

Usage:
    uv run phase1/inspect_day_schedule.py
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

from backtest.engine import (
    DEFAULT_BATTERY,
    DEFAULT_OPT_SETTINGS,
    HORIZON,
    SETTLE_PERIODS,
    _normalise_to_30min,
)
from optimiser.model import build_model
from optimiser.scenarios import sample_scenarios_multimarket, _estimate_correlation_matrix
from run_backtest import _load_actual, _load_dc_actual, _load_forecast

from run_fixed_month_comparison import make_perfect_foresight_builder
from historical_error_scenarios import build_error_path_pool, make_historical_error_scenario_builder

# ---------------------------------------------------------------------------
# Target day — perfect_foresight's worst DA day from the Dec 2022 comparison
# ---------------------------------------------------------------------------
TARGET_DATE = "2022-12-15"
PERIOD = "cv"          # Dec 2022 falls in the CV/OOF forecast period
N_INSPECT_SCENARIOS = 50  # match --n-scenarios used in the comparison run

# current_soc going into this day: ideally read off the previous day's
# ending SOC from a full run, but for a one-off inspection like this,
# starting from the battery's default SOC is a reasonable approximation —
# the goal here is to see *when* DA charging happens relative to BM revenue,
# not to reproduce the exact daily total to the penny. Same approximation
# applies to both strategies below, so the comparison between them is still
# fair even though neither exactly matches its full-month-run total.
STARTING_SOC = DEFAULT_BATTERY["current_soc"]


def _solve_and_extract(model, battery_id, da_win, bm_win, window_start, label):
    """Solve a built model and extract the settled half as a labeled DataFrame."""
    solver = SolverFactory("appsi_highs")  # adjust to match run_backtest's actual solver
    solver.options["time_limit"] = 120
    result = solver.solve(model)

    if result.solver.termination_condition != TerminationCondition.optimal:
        print(f"[{label}] Solve failed: {result.solver.termination_condition}")
        return None

    idx = pd.date_range(window_start, periods=SETTLE_PERIODS, freq="30min", tz=window_start.tzinfo)
    da_vals = da_win["value"].iloc[:SETTLE_PERIODS].to_numpy()
    bm_vals = bm_win["value"].iloc[:SETTLE_PERIODS].to_numpy()

    rows = []
    for t in range(SETTLE_PERIODS):
        da_d = pyo.value(model.da_discharge[t, battery_id])
        da_c = pyo.value(model.da_charge[t, battery_id])
        bm_mw = pyo.value(model.bm_offer[t, battery_id])
        soc = pyo.value(model.soc[t, battery_id])

        rows.append({
            "datetime": idx[t],
            "da_price": da_vals[t],
            "bm_price": bm_vals[t],
            "da_discharge_mw": round(da_d, 4),
            "da_charge_mw": round(da_c, 4),
            "bm_offer_mw": round(bm_mw, 4),
            "soc": round(soc, 4),
        })

    return pd.DataFrame(rows).set_index("datetime")


def main() -> None:
    ref_tz = "UTC"
    window_start = pd.Timestamp(TARGET_DATE, tz=ref_tz)
    window_end = window_start + timedelta(hours=48)

    # ------------------------------------------------------------------
    # Load actuals, normalised exactly like run_backtest expects
    # ------------------------------------------------------------------
    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    bm_actual = _load_actual("processed/bmrs_processed.parquet", "systemSellPrice").to_frame("value")
    da_actual_30 = _normalise_to_30min(da_actual, "da_actual")

    dc_low_raw  = _load_dc_actual("DCL")
    dc_high_raw = _load_dc_actual("DCH")
    dc_low_30  = dc_low_raw.resample("30min").ffill().to_frame("value")
    dc_high_30 = dc_high_raw.resample("30min").ffill().to_frame("value")

    da_win  = da_actual_30.loc[window_start : window_end - timedelta(minutes=30)]
    bm_win  = bm_actual.loc[window_start : window_end - timedelta(minutes=30)]
    dcl_win = dc_low_30.loc[window_start : window_end - timedelta(minutes=30)]
    dch_win = dc_high_30.loc[window_start : window_end - timedelta(minutes=30)]

    # ------------------------------------------------------------------
    # Load forecasts for this window (needed for current_scenarios)
    # ------------------------------------------------------------------
    da_fc  = _load_forecast("da",      PERIOD)
    bm_fc  = _load_forecast("bm",      PERIOD)
    dcl_fc = _load_forecast("dc_low",  PERIOD)
    dch_fc = _load_forecast("dc_high", PERIOD)

    da_fc_win  = da_fc.loc[window_start : window_end - timedelta(minutes=30)].iloc[:HORIZON]
    bm_fc_win  = bm_fc.loc[window_start : window_end - timedelta(minutes=30)].iloc[:HORIZON]
    dcl_fc_win = dcl_fc.loc[window_start : window_end - timedelta(minutes=30)].iloc[:HORIZON]
    dch_fc_win = dch_fc.loc[window_start : window_end - timedelta(minutes=30)].iloc[:HORIZON]

    battery_id = DEFAULT_BATTERY["id"]
    battery = dict(DEFAULT_BATTERY)
    battery["current_soc"] = STARTING_SOC

    # ------------------------------------------------------------------
    # Case 1: perfect_foresight
    # ------------------------------------------------------------------
    pf_builder = make_perfect_foresight_builder(da_actual_30, bm_actual, dc_low_30, dc_high_30)
    pf_scenarios = pf_builder(window_start, HORIZON, None, None, None, None, seed=42)
    pf_model = build_model([dict(battery)], pf_scenarios, DEFAULT_OPT_SETTINGS, window_start=window_start)
    pf_schedule = _solve_and_extract(pf_model, battery_id, da_win, bm_win, window_start, "perfect_foresight")

    # ------------------------------------------------------------------
    # Case 2: current_scenarios — same approach run_backtest uses when
    # scenario_builder=None: estimate the correlation matrix as of this
    # date, then sample the Gaussian copula scenarios from the forecasts
    # ------------------------------------------------------------------
    corr_matrix = _estimate_correlation_matrix(cutoff_date=str(window_start.date()))
    cs_scenarios = sample_scenarios_multimarket(
        da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win,
        n=N_INSPECT_SCENARIOS,
        seed=42,
        corr_matrix=corr_matrix,
    )
    cs_model = build_model([dict(battery)], cs_scenarios, DEFAULT_OPT_SETTINGS, window_start=window_start)
    cs_schedule = _solve_and_extract(cs_model, battery_id, da_win, bm_win, window_start, "current_scenarios")

    # ------------------------------------------------------------------
    # Case 3: historical_error_scenarios (Phase 2) — build the pool from
    # the full CV period, then sample joint historical error days for
    # this specific window.
    # ------------------------------------------------------------------
    error_pool = build_error_path_pool(
        da_actual_30, bm_actual, dc_low_30, dc_high_30,
        da_fc, bm_fc, dcl_fc, dch_fc,
    )
    he_builder = make_historical_error_scenario_builder(error_pool, n=N_INSPECT_SCENARIOS)
    he_scenarios = he_builder(window_start, HORIZON, da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win, seed=42)
    he_model = build_model([dict(battery)], he_scenarios, DEFAULT_OPT_SETTINGS, window_start=window_start)
    he_schedule = _solve_and_extract(he_model, battery_id, da_win, bm_win, window_start, "historical_error_scenarios")

    # ------------------------------------------------------------------
    # Seed sensitivity check: is Dec-15 passivity a property of the method,
    # or an unlucky draw of 20 historical days for this specific window?
    # Re-solve with several different seeds and compare total settled
    # da_charge / bm_offer magnitude across the day for each.
    # ------------------------------------------------------------------
    print("\n=== seed sensitivity: historical_error_scenarios on 2022-12-15 ===")
    for seed in [0, 1, 7, 42, 99, 123]:
        seed_scenarios = he_builder(window_start, HORIZON, da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win, seed=seed)
        seed_model = build_model([dict(battery)], seed_scenarios, DEFAULT_OPT_SETTINGS, window_start=window_start)
        seed_schedule = _solve_and_extract(
            seed_model, battery_id, da_win, bm_win, window_start, f"he_seed_{seed}"
        )
        if seed_schedule is None:
            print(f"  seed={seed:4d}  SOLVE FAILED")
            continue
        total_da_charge = seed_schedule["da_charge_mw"].sum()
        total_da_disch  = seed_schedule["da_discharge_mw"].sum()
        total_bm_offer  = seed_schedule["bm_offer_mw"].sum()
        n_active_periods = ((seed_schedule["da_charge_mw"].abs() > 0.01) |
                             (seed_schedule["da_discharge_mw"].abs() > 0.01) |
                             (seed_schedule["bm_offer_mw"].abs() > 0.01)).sum()
        print(
            f"  seed={seed:4d}  total_da_charge={total_da_charge:6.2f}  "
            f"total_da_discharge={total_da_disch:6.2f}  total_bm_offer={total_bm_offer:6.2f}  "
            f"active_periods={n_active_periods}/48"
        )

    # ------------------------------------------------------------------
    # Print and save both, side by side
    # ------------------------------------------------------------------
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    if pf_schedule is not None:
        print("\n=== perfect_foresight ===")
        print(pf_schedule)
        pf_schedule.to_csv(f"data/results/phase1_dec2022/schedule_perfect_foresight_{TARGET_DATE}.csv")

    if cs_schedule is not None:
        print("\n=== current_scenarios ===")
        print(cs_schedule)
        cs_schedule.to_csv(f"data/results/phase1_dec2022/schedule_current_scenarios_{TARGET_DATE}.csv")

    if he_schedule is not None:
        print("\n=== historical_error_scenarios ===")
        print(he_schedule)
        he_schedule.to_csv(f"data/results/phase1_dec2022/schedule_historical_error_scenarios_{TARGET_DATE}.csv")

    if pf_schedule is not None and cs_schedule is not None and he_schedule is not None:
        compare = pd.DataFrame({
            "da_price": pf_schedule["da_price"],
            "bm_price": pf_schedule["bm_price"],
            "pf_da_charge": pf_schedule["da_charge_mw"],
            "cs_da_charge": cs_schedule["da_charge_mw"],
            "he_da_charge": he_schedule["da_charge_mw"],
            "pf_bm_offer": pf_schedule["bm_offer_mw"],
            "cs_bm_offer": cs_schedule["bm_offer_mw"],
            "he_bm_offer": he_schedule["bm_offer_mw"],
        })
        print("\n=== side-by-side comparison (all three) ===")
        print(compare)
        compare.to_csv(f"data/results/phase1_dec2022/schedule_comparison_{TARGET_DATE}.csv")
        print(f"\nSaved comparison to: data/results/phase1_dec2022/schedule_comparison_{TARGET_DATE}.csv")


if __name__ == "__main__":
    main()