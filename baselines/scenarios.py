"""
baselines/scenarios.py — Deterministic single-scenario builders for baseline strategies.

Each function returns a scenario dict with the same structure as
`sample_scenarios_multimarket`, but with n=1 and no copula sampling:

    {
        "da":      np.ndarray, shape (1, horizon)
        "bm":      np.ndarray, shape (1, horizon)
        "dc_low":  np.ndarray, shape (1, horizon)
        "dc_high": np.ndarray, shape (1, horizon)
        "seed":    seed value or None
    }

This means every baseline passes through the same `build_model`, solver,
`settle_revenue`, SOC carryover, and `da_imputed` join as the main strategy.
The only thing that changes is the forecast assumption.

Callable signatures
-------------------
All builders and factories produce callables with this signature so they can
be passed as `scenario_builder` to `run_backtest`:

    fn(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win, seed)

`window_start` is a tz-aware pd.Timestamp for the start of the 48-hour window.
`horizon` is the number of 30-min periods in the optimiser horizon (normally 96).
The fc_win arguments are DataFrames with q10/q50/q90 columns, pre-sliced to the window.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(df: pd.DataFrame) -> pd.Series:
    """Return the price column from an actual-price DataFrame."""
    return df["value"] if "value" in df.columns else df.iloc[:, 0]


def _block_avg_1d(arr: np.ndarray, horizon: int) -> np.ndarray:
    """
    Average within each 4-hour EFA block and tile back to half-hourly resolution.

    Ensures DC scenario prices are block-constant, matching the market structure
    where a single clearing price applies to all 8 periods within a block.
    Requires horizon % 8 == 0 (enforced by the horizon guard in scenarios.py).
    """
    n_blocks = horizon // 8
    return np.repeat(arr.reshape(n_blocks, 8).mean(axis=1), 8)


# ---------------------------------------------------------------------------
# Median baseline
# ---------------------------------------------------------------------------


def build_median_scenario(
    window_start: pd.Timestamp,
    horizon: int,
    da_fc_win: pd.DataFrame,
    bm_fc_win: pd.DataFrame,
    dc_low_fc_win: pd.DataFrame,
    dc_high_fc_win: pd.DataFrame,
    seed=None,
) -> dict:
    """
    Single-scenario from q50 quantile forecasts.

    DA and BM use the raw q50 path (period-by-period). DC Low and DC High
    are block-averaged before use, consistent with the main scenario generator
    (DC prices clear at EFA block level; a block-constant price is more
    representative than a path that varies within the block).

    This is the fairest deterministic comparison — it uses the same information
    as the main model (the trained forecasters) but removes all uncertainty
    quantification, running the optimiser as if the q50 were the certain outcome.
    """
    if horizon % 8 != 0:
        raise ValueError(
            f"horizon={horizon} must be divisible by 8 for DC block averaging."
        )

    da_q50  = da_fc_win["q50"].values[:horizon]
    bm_q50  = bm_fc_win["q50"].values[:horizon]
    dcl_q50 = _block_avg_1d(dc_low_fc_win["q50"].values[:horizon],  horizon)
    dch_q50 = _block_avg_1d(dc_high_fc_win["q50"].values[:horizon], horizon)

    return {
        "da":      da_q50[np.newaxis, :],
        "bm":      bm_q50[np.newaxis, :],
        "dc_low":  dcl_q50[np.newaxis, :],
        "dc_high": dch_q50[np.newaxis, :],
        "seed":    seed,
    }


# ---------------------------------------------------------------------------
# Naive lag baseline factory
# ---------------------------------------------------------------------------


def make_naive_scenario_builder(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual: pd.DataFrame,
    dc_high_actual: pd.DataFrame,
    lag_periods: int,
) -> callable:
    """
    Return a scenario_builder for the naive lag baseline.

    For each backtest step at window_start, the "forecast" for period t is the
    realised price at (window_start - lag_offset + t * 30min), where
    lag_offset = lag_periods * 30 minutes.

    lag_periods=48  → previous-day baseline (24 hours back)
    lag_periods=336 → previous-week baseline (168 hours back)

    The returned callable has the standard scenario_builder signature so it
    can be passed directly to run_backtest as `scenario_builder=...`.

    All four actual DataFrames must be indexed by a tz-aware DatetimeIndex at
    30-minute resolution (as produced by run_backtest's normalisation step).
    Pass the same DataFrames you would pass to run_backtest as actuals.

    NaN handling: if the lag window contains NaN values (data gaps or the
    window falls before the start of available data), the affected periods are
    forward-filled within the window using the last valid observation. If the
    entire window is NaN a warning is logged and zeros are used — the solve
    will likely return poor results and those days should already be flagged
    by the data_quality audit.
    """
    lag_offset = pd.Timedelta(minutes=30 * lag_periods)
    label = f"lag_{lag_periods}"

    def _build(
        window_start: pd.Timestamp,
        horizon: int,
        da_fc_win: pd.DataFrame,
        bm_fc_win: pd.DataFrame,
        dc_low_fc_win: pd.DataFrame,
        dc_high_fc_win: pd.DataFrame,
        seed=None,
    ) -> dict:
        if horizon % 8 != 0:
            raise ValueError(
                f"horizon={horizon} must be divisible by 8 for DC block averaging."
            )

        # The lag window starts lag_offset before the current optimisation window
        lag_start = window_start - lag_offset
        lag_index = pd.date_range(
            start=lag_start,
            periods=horizon,
            freq="30min",
            tz=window_start.tzinfo,
        )

        def _extract(df: pd.DataFrame, name: str) -> np.ndarray:
            series = _col(df).reindex(lag_index)
            n_nan = series.isna().sum()
            if n_nan > 0:
                if n_nan == len(series):
                    logger.warning(
                        "%s [%s]: entire lag window at %s is NaN — "
                        "falling back to zeros. Check data coverage.",
                        label, name, window_start.date(),
                    )
                    return np.zeros(horizon)
                logger.debug(
                    "%s [%s]: %d NaN period(s) in lag window at %s — forward-filling.",
                    label, name, n_nan, window_start.date(),
                )
                series = series.ffill()
            return series.to_numpy(dtype=float, na_value=np.nan)

        da_vals  = _extract(da_actual,       "DA")
        bm_vals  = _extract(bm_actual,       "BM")
        dcl_vals = _extract(dc_low_actual,   "DC_Low")
        dch_vals = _extract(dc_high_actual,  "DC_High")

        # DC actual prices already clear at block level (forward-filled EFA blocks
        # in the pipeline), but block-average for safety to ensure strict block
        # constancy in the scenario — same treatment as build_median_scenario.
        dcl_vals = _block_avg_1d(dcl_vals, horizon)
        dch_vals = _block_avg_1d(dch_vals, horizon)

        return {
            "da":      da_vals[np.newaxis, :],
            "bm":      bm_vals[np.newaxis, :],
            "dc_low":  dcl_vals[np.newaxis, :],
            "dc_high": dch_vals[np.newaxis, :],
            "seed":    seed,
        }

    _build.__name__ = f"naive_scenario_builder_{label}"
    return _build

