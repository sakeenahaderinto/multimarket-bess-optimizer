"""
Multi-service BESS backtesting engine.

Walks forward day by day through historical data, running the Pyomo MIP
optimiser for each 48-hour horizon using pre-trained forecast quantiles
as scenario inputs. Realised revenue is settled against actual market prices.
"""

import logging
from datetime import timedelta
import copy

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from config import settings
from optimiser.model import build_model
from optimiser.scenarios import sample_scenarios_multimarket
from optimiser.scenarios import _estimate_correlation_matrix


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default battery spec — matches the demo notebook
# ---------------------------------------------------------------------------
DEFAULT_BATTERY = {
    "id": "bat_01",
    "capacity_mwh": 1.0,
    "max_power_mw": 1.0,
    "current_soc": 0.5,
    "round_trip_efficiency": 0.85,
    "design_cycle_life": 4000,
}

DEFAULT_OPT_SETTINGS = {
    "replacement_cost_per_mwh": "175000",
    "dc_efa_block_min_hours": "4",
    #"dc_soc_min": "0.4",
    #"dc_soc_max": "0.6",
    "dc_max_fraction": "0.5",
    "dc_response_duration_h": "0.5",
    "terminal_soc_min": "0.1",
}

HORIZON = 96  # 48 hours x 2 periods/hour: full optimizer horizon
SETTLE_PERIODS = 48 # 24 hours x 2 periods/hour: settiled each step. 
                    # NOTE: In the Pyomo model: T_SOC = range(97),
                    #  soc[0]=initial, soc[t]=SOC after period t-1. 
                    # So soc[48] = SOC after period 47 = end of first 24 hours
N_SCENARIOS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_tz(df: pd.DataFrame, name: str) -> str | None:
    """Return the tzinfo of a DataFrame's DatetimeIndex, or None if naive."""
    tz = getattr(df.index, "tz", None)
    if tz is None:
        logger.warning(
            "Input '%s' has a timezone-naive index — will treat as UTC. "
            "Pass tz-aware data to avoid silent misalignment.",
            name,
        )
    return tz


def _normalise_to_30min(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Resample a DataFrame to 30-min frequency using forward-fill.

    Uses inferred frequency rather than relying on index.freq, which is
    commonly None after slicing, merging, or loading from CSV.
    """
    if df.empty:
        return df
    inferred = pd.infer_freq(df.index)
    if inferred == "30min" or inferred == "30T":
        return df
    logger.debug(
        "Resampling '%s' from inferred freq '%s' to 30min.", name, inferred
    )
    return df.resample("30min").ffill()


# ---------------------------------------------------------------------------
# Revenue settlement against actual prices
# ---------------------------------------------------------------------------


def settle_revenue(
    model: pyo.ConcreteModel,
    da_actual: pd.Series,
    bm_actual: pd.Series,
    dc_low_actual: pd.Series,
    dc_high_actual: pd.Series,
    battery_id: str,
    n_periods: int,
    battery: dict | None = None,
    opt_settings: dict | None = None,
) -> dict:
    """
    Calculate realised revenue using actual market prices and the
    dispatch schedule produced by the optimiser.

    DA and BM revenue: price (£/MWh) x power (MW) x 0.5h = £
    DC revenue: clearing price (£/MW/h) x reserved MW x 0.5h = £

    Actual series must be pre-aligned to the dispatch window with exactly n_periods rows.
    Settlement uses positional indexing into .values — series must be gap-free and ordered.

    BM settlement is an expected-value approximation: bm_offer (scenario-dependent) is
    averaged across scenarios and settled against the single realised price. This is not
    a physical dispatch path — see session_fixes_and_decisions.md S2-11 for implications.
    """
    if battery is None:
        battery = DEFAULT_BATTERY
    if opt_settings is None:
        opt_settings = DEFAULT_OPT_SETTINGS

    da_rev = bm_rev = dcl_rev = dch_rev = deg = 0.0

    rep_cost = float(opt_settings.get("replacement_cost_per_mwh", 175_000))
    cycle_life = battery.get("design_cycle_life", DEFAULT_BATTERY["design_cycle_life"])
    kappa = rep_cost / (2 * cycle_life)  # £/MWh throughput

    n_s = len(model.S)

    
    da_vals = da_actual.values
    bm_vals = bm_actual.values
    dcl_vals = dc_low_actual.values
    dch_vals = dc_high_actual.values

    for t in range(n_periods):
        da_d = pyo.value(model.da_discharge[t, battery_id])
        da_c = pyo.value(model.da_charge[t, battery_id])


        da_p = float(da_vals[t]) if t < len(da_vals) else 0.0
        da_rev += da_p * (da_d - da_c) * 0.5

        bm_p = float(bm_vals[t]) if t < len(bm_vals) else 0.0

        
        bm_mw = sum(
            pyo.value(model.bm_offer[t, battery_id, s]) for s in range(n_s)
        ) / n_s
        bm_rev += bm_p * bm_mw * 0.5

        dcl_p = float(dcl_vals[t]) if t < len(dcl_vals) else 0.0
        dch_p = float(dch_vals[t]) if t < len(dch_vals) else 0.0
        dcl_mw = pyo.value(model.dc_low[t, battery_id])
        dch_mw = pyo.value(model.dc_high[t, battery_id])
        dcl_rev += dcl_p * dcl_mw * 0.5
        dch_rev += dch_p * dch_mw * 0.5

        # Degradation
        deg += (da_d + da_c) * 0.5 * kappa
        deg += bm_mw * 0.5 * kappa

    gross = da_rev + bm_rev + dcl_rev + dch_rev
    net = gross - deg

    return {
        "da_revenue": round(da_rev, 2),
        "bm_revenue": round(bm_rev, 2),
        "dc_low_revenue": round(dcl_rev, 2),
        "dc_high_revenue": round(dch_rev, 2),
        "gross_revenue": round(gross, 2),
        "degradation_cost": round(deg, 2),
        "net_revenue": round(net, 2),
    }


def _read_ending_soc(model: pyo.ConcreteModel, battery_id: str, settle_t: int | None = None) -> float | None:
    """
    Extract the SOC at the end of the settled horizon from a solved model.
    settle_t: the T_SOC index at the settlement boundary (e.g. 48 for end of day 1).
    Defaults to the final period of the full horizon if not specified.
    Returns None if the variable is not present or has no value.

    settle_t: T_SOC index at the settlement boundary. For SETTLE_PERIODS=48,
    this is soc[48] = SOC after dispatch period 47 = end of the first 24 hours.
    T_SOC is indexed 0..n_periods: soc[0] is the initial SOC, soc[t] is the
    SOC after period t-1. An off-by-one here would silently shift every
    carried-forward SOC by one half-hour — verify if SETTLE_PERIODS changes.
    Returns None if the variable is not present or has no value.
    """

    try:
        target_t = settle_t if settle_t is not None else max(t for (t, b, s) in model.soc.keys() if b == battery_id)
        soc_vals = [
            pyo.value(model.soc[target_t, battery_id, s])
            for (t, b, s) in model.soc.keys()
            if t == target_t and b == battery_id
        ]
        return sum(soc_vals) / len(soc_vals)
    except (AttributeError, ValueError, ZeroDivisionError):
        return None



# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_backtest(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual: pd.DataFrame,
    dc_high_actual: pd.DataFrame,
    da_forecast: pd.DataFrame,
    bm_forecast: pd.DataFrame,
    dc_low_forecast: pd.DataFrame,
    dc_high_forecast: pd.DataFrame,
    battery: dict | None = None,
    opt_settings: dict | None = None,
    step_days: int = 1,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
    scenario_builder=None,
    spread_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:

    """
    Walk-forward backtest over historical data.

    At each step:
      1. Slice 48-hour forecast window
      2. Generate N_SCENARIOS price scenarios
      3. Solve the MIP optimiser
      4. Settle revenue against actual prices
      5. Carry ending SOC forward as starting SOC for the next step
      6. Record results

    Args:
        da_actual:   Actual DA prices, 30-min index, column 'value' (£/MWh)
        bm_actual:   Actual BM system sell prices, 30-min index (£/MWh)
        dc_low_actual:  Actual DC Low clearing prices, EFA-block or 30-min index (£/MW/h).
                        Forward-filled to 30-min internally if at lower resolution.
        dc_high_actual: Actual DC High clearing prices, EFA-block or 30-min index (£/MW/h).
                        Forward-filled to 30-min internally if at lower resolution.
        da_forecast:    DA forecast with q10/q50/q90 columns
        bm_forecast:    BM forecast with q10/q50/q90 columns
        dc_low_forecast:  DC Low forecast with q10/q50/q90 columns
        dc_high_forecast: DC High forecast with q10/q50/q90 columns
        battery:      Battery spec dict (defaults to DEFAULT_BATTERY)
        opt_settings: Optimiser settings dict
        step_days:    How many days to advance per backtest step
        start_date:   Optional start date string 'YYYY-MM-DD'
        end_date:     Optional end date string 'YYYY-MM-DD'
        seed:         Random seed for scenario generation
        scenario_builder: Optional callable for baseline strategies. If provided,
            it replaces sample_scenarios_multimarket. Signature:
            fn(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win,
               dc_high_fc_win, seed) -> dict with keys "da", "bm", "dc_low",
            "dc_high" each shape (n_scenarios, horizon). The correlation matrix
            estimation step is also skipped when this is set.
    Returns:
        DataFrame with one row per backtest step and revenue breakdown columns.
    """
    if battery is None:
        battery = DEFAULT_BATTERY.copy()
    if opt_settings is None:
        opt_settings = DEFAULT_OPT_SETTINGS.copy()

    if step_days != 1:
        raise ValueError(
            f"step_days={step_days} is not supported. SETTLE_PERIODS={SETTLE_PERIODS} assumes "
            "daily steps (step_days=1). Update SETTLE_PERIODS = step_days * 48 if changing step size."
        )


    # Detect the timezone from da_actual and coerce everything to match.
    input_frames = {
        "da_actual": da_actual,
        "bm_actual": bm_actual,
        "dc_low_actual": dc_low_actual,
        "dc_high_actual": dc_high_actual,
        "da_forecast": da_forecast,
        "bm_forecast": bm_forecast,
        "dc_low_forecast": dc_low_forecast,
        "dc_high_forecast": dc_high_forecast,
    }
    if spread_forecast is not None:
        input_frames["spread_forecast"] = spread_forecast
    for name, df in input_frames.items():
        _infer_tz(df, name)


    ref_tz = getattr(da_actual.index, "tz", None)
    if ref_tz is None:
        ref_tz = "UTC"
        # Localise all naive indexes to UTC
        for name, df in input_frames.items():
            if getattr(df.index, "tz", None) is None:
                df.index = df.index.tz_localize("UTC")
    else:
        # Convert any frames that differ from the reference timezone
        for name, df in input_frames.items():
            frame_tz = getattr(df.index, "tz", None)
            if frame_tz is None:
                df.index = df.index.tz_localize(ref_tz)
            elif str(frame_tz) != str(ref_tz):
                logger.warning(
                    "Input '%s' has tz=%s; converting to %s.", name, frame_tz, ref_tz
                )
                df.index = df.index.tz_convert(ref_tz)
    da_actual = _normalise_to_30min(da_actual, "da_actual")
    dc_low_actual = _normalise_to_30min(dc_low_actual, "dc_low_actual")
    dc_high_actual = _normalise_to_30min(dc_high_actual, "dc_high_actual")


    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = 120  # seconds per solve

    # Determine backtest date range
    all_dates = sorted(set(da_actual.index.date))
    if start_date:
        all_dates = [d for d in all_dates if str(d) >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if str(d) <= end_date]

    # Step by step_days
    step_dates = all_dates[::step_days]

    if not step_dates:
        logger.warning(
            "Backtest: no dates in range (start=%s end=%s) — returning empty.",
            start_date,
            end_date,
        )
        return pd.DataFrame()

    results = []
    n_solved = 0
    n_failed = 0


    current_soc = battery.get("current_soc", DEFAULT_BATTERY["current_soc"])

    logger.info(
        "Backtest: %d steps from %s to %s",
        len(step_dates),
        step_dates[0],
        step_dates[-1],
    )

    

    for i, current_date in enumerate(step_dates):
        # modulo log for long runs to avoid flooding.
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(step_dates):
            logger.info(
                "  Step %d / %d  (%s)  solved=%d failed=%d",
                i + 1,
                len(step_dates),
                current_date,
                n_solved,
                n_failed,
            )

        # 48-hour window starting at current_date
        window_start = pd.Timestamp(current_date, tz=ref_tz)
        window_end = window_start + timedelta(hours=48)

        # Slice actual prices for this window
        da_win = da_actual.loc[window_start : window_end - timedelta(minutes=30)]
        bm_win = bm_actual.loc[window_start : window_end - timedelta(minutes=30)]
        dcl_win = dc_low_actual.loc[window_start : window_end - timedelta(minutes=30)]
        dch_win = dc_high_actual.loc[window_start : window_end - timedelta(minutes=30)]

        # Slice forecasts for this window
        da_fc_win = da_forecast.loc[window_start : window_end - timedelta(minutes=30)]
        bm_fc_win = bm_forecast.loc[window_start : window_end - timedelta(minutes=30)]
        dcl_fc_win = dc_low_forecast.loc[
            window_start : window_end - timedelta(minutes=30)
        ]
        dch_fc_win = dc_high_forecast.loc[
            window_start : window_end - timedelta(minutes=30)
        ]
        spread_fc_win = (
            spread_forecast.loc[window_start : window_end - timedelta(minutes=30)]
            if spread_forecast is not None else None
        )

        data_windows = {
            "da_actual": da_win,
            "bm_actual": bm_win,
            "dc_low_actual": dcl_win,
            "dc_high_actual": dch_win,
            "da_fc": da_fc_win,
            "bm_fc": bm_fc_win,
            "dc_low_fc": dcl_fc_win,
            "dc_high_fc": dch_fc_win,
        }
        if spread_fc_win is not None:
            data_windows["spread_fc"] = spread_fc_win
        short = [k for k, s in data_windows.items()
                 if (k.endswith("_actual") and len(s) < SETTLE_PERIODS)
                 or (k.endswith("_fc") and len(s) < HORIZON)]
        if short:
            logger.warning(
                "Skipping %s — insufficient data for: %s (got lengths: %s)",
                current_date,
                short,
                {k: len(data_windows[k]) for k in short},
            )
            continue


        # Align forecast to HORIZON periods
        da_fc_win = da_fc_win.iloc[:HORIZON]
        bm_fc_win = bm_fc_win.iloc[:HORIZON]
        dcl_fc_win = dcl_fc_win.iloc[:HORIZON]
        dch_fc_win = dch_fc_win.iloc[:HORIZON]
        if spread_fc_win is not None:
            spread_fc_win = spread_fc_win.iloc[:HORIZON]

        try:
            if scenario_builder is None:
                corr_matrix = _estimate_correlation_matrix(cutoff_date=str(current_date),
                                                           use_spread=spread_fc_win is not None,)

                scenarios = sample_scenarios_multimarket(
                    da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win,
                    n=N_SCENARIOS,
                    seed=seed + i,
                    corr_matrix=corr_matrix,
                    spread_fc=spread_fc_win,
                )
            else:
                scenarios = scenario_builder(
                    window_start, HORIZON,
                    da_fc_win, bm_fc_win, dcl_fc_win, dch_fc_win,
                    seed + i,
                )



            for key in ("da", "bm", "dc_low", "dc_high"):
                arr = scenarios.get(key)
                if arr is not None and np.isnan(np.asarray(arr, dtype=float)).any():
                    raise ValueError(
                        f"NaN scenario prices for '{key}' on {current_date} — "
                        "check forecast coverage for this date."
                    )

            step_battery = copy.deepcopy(battery)
            step_battery["current_soc"] = current_soc

            batteries = [step_battery]
            model = build_model(batteries, scenarios, opt_settings)
            result = solver.solve(model)

            if result.solver.termination_condition != TerminationCondition.optimal:
                logger.warning(
                    "Solve failed for %s: %s",
                    current_date,
                    result.solver.termination_condition,
                )
                n_failed += 1

                results.append(
                    {
                        "date": current_date,
                        "solve_failed": True,
                        "da_revenue": float("nan"),
                        "bm_revenue": float("nan"),
                        "dc_low_revenue": float("nan"),
                        "dc_high_revenue": float("nan"),
                        "gross_revenue": float("nan"),
                        "degradation_cost": float("nan"),
                        "net_revenue": float("nan"),
                    }
                )
                del model
                continue

            # Settle revenue against actual prices
            revenue = settle_revenue(
                model,
                (da_win["value"]  if "value"  in da_win.columns  else da_win.iloc[:,  0]).iloc[:SETTLE_PERIODS],
                (bm_win["value"]  if "value"  in bm_win.columns  else bm_win.iloc[:,  0]).iloc[:SETTLE_PERIODS],
                (dcl_win["value"] if "value" in dcl_win.columns else dcl_win.iloc[:, 0]).iloc[:SETTLE_PERIODS],
                (dch_win["value"] if "value" in dch_win.columns else dch_win.iloc[:, 0]).iloc[:SETTLE_PERIODS],
                battery["id"],
                SETTLE_PERIODS,
                battery=step_battery,
                opt_settings=opt_settings,
            )
            revenue["date"] = current_date
            revenue["solve_failed"] = False
            results.append(revenue)
            n_solved += 1

            ending_soc = _read_ending_soc(model, battery["id"], settle_t=SETTLE_PERIODS)
            if ending_soc is not None:
                current_soc = ending_soc
            else:
                logger.warning(
                    "Could not read ending SOC for %s; keeping previous SOC=%.3f.",
                    current_date,
                    current_soc,
                )


            del model

        except Exception as e:
            logger.error("Error on %s: %s", current_date, e, exc_info=True)
            n_failed += 1
            results.append(
                {
                    "date": current_date,
                    "solve_failed": True,
                    "da_revenue": float("nan"),
                    "bm_revenue": float("nan"),
                    "dc_low_revenue": float("nan"),
                    "dc_high_revenue": float("nan"),
                    "gross_revenue": float("nan"),
                    "degradation_cost": float("nan"),
                    "net_revenue": float("nan"),
                }
            )



    logger.info("Backtest complete: solved=%d  failed=%d", n_solved, n_failed)

    if not results:
        logger.warning("No results — check date range and data availability.")
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("date")

    quality_path = settings.data_dir / "quality" / "da_imputed_by_date.parquet"
    if quality_path.exists():
        imputed_flags = pd.read_parquet(quality_path)[["da_imputed"]]
        imputed_flags.index = pd.to_datetime(imputed_flags.index).date
        df["da_imputed"] = df.index.map(imputed_flags["da_imputed"]).fillna(False)
    else:
        logger.warning(
            "DA imputation flag not found at %s — run data_quality.py to generate it. "
            "Backtest results will not include 'da_imputed' column.",
            quality_path,
        )

    return df
