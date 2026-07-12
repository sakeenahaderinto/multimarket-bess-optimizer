"""
historical_error_scenarios.py

Historical forecast-error-path scenario generator. Addresses the limitation
of per-period-independent Gaussian coupla draws, which cannot reproduce the
cross-market structure of real price deviations

Method
------
For each historical day with complete OOF forecasts and actuals across all
four markets:
    error[market, t] = actual[market, t] - q50_forecast[market, t]
This gives one joint, 48-period, 4-market error *path* per historical day --
a real realisation of how all four markets jointly deviated from their
central forecast on that specific day, preserving both the temporal shape
within the day and the cross-market relationship, by construction (no
estimated correlation matrix is used at all).

To build a scenario for a new target window, sample n historical days'
error paths (with replacement) and add each one, period-by-period, to the
target day's own q50 forecast:
    scenario[market, t] = q50_forecast_target[market, t] + error_pool[market, t]

DC Low/High errors are computed and added at native EFA-block resolution
(after block-averaging both the actual and the q50 forecast), then the
resulting scenario is block-averaged again for strict consistency with how
every other scenario builder in this codebase treats DC.

The builder filters the pool to days before window_start.date() on each call
to prevent future error leakage

Dates excluded via excluded_dates.is_excluded() (clock-change days, BM's six
genuine data-gap anomaly days) are never used as a source for sampling.
"""

import logging

import numpy as np
import pandas as pd

from baselines.scenarios import _col, _block_avg_1d
from optimiser.scenarios import _efa_block_groups
from excluded_dates import is_excluded

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: build the historical error-path pool
# ---------------------------------------------------------------------------

def _compute_da_vol_7d(da_actual: pd.DataFrame, dates: list) -> np.ndarray:
    """7-day trailing DA price std for each date, using only data before that date."""
    vols = []
    for d in dates:
        lookback = pd.date_range(
            end=pd.Timestamp(d, tz="UTC") - pd.Timedelta(minutes=30),
            periods=7 * 48,
            freq="30min",
        )
        window = _col(da_actual).reindex(lookback).dropna()
        vols.append(window.std() if len(window) >= 48 else np.nan)
    return np.array(vols)


def build_error_path_pool(
    da_actual: pd.DataFrame,
    bm_actual: pd.DataFrame,
    dc_low_actual_30: pd.DataFrame,
    dc_high_actual_30: pd.DataFrame,
    da_oof: pd.DataFrame,
    bm_oof: pd.DataFrame,
    dc_low_oof: pd.DataFrame,
    dc_high_oof: pd.DataFrame,
) -> dict:
    """
    Build the pool of complete historical joint error days.

    Inputs:
        *_actual:  realised prices, 30-min DatetimeIndex, column 'value'
                   (da_actual, bm_actual already normalised to 30-min;
                   dc_*_actual_30 already block-forward-filled to 30-min).
        *_oof:     out-of-fold forecasts with q10/q50/q90 columns, 30-min
                   DatetimeIndex, covering the same period as the actuals.

    Outputs:
        dict with:
            "dates":    np.ndarray of valid source dates (date objects)
            "da":       np.ndarray, shape (n_valid_days, 48)
            "bm":       np.ndarray, shape (n_valid_days, 48)
            "dc_low":   np.ndarray, shape (n_valid_days, 48)
            "dc_high":  np.ndarray, shape (n_valid_days, 48)
        Each row is one historical day's joint error path. dc_low/dc_high
        errors are pre-block-averaged (8-period blocks), consistent with
        every other DC treatment in this codebase.
    """
    da_err  = (_col(da_actual)  - da_oof["q50"]).dropna()
    bm_err  = (_col(bm_actual)  - bm_oof["q50"]).dropna()
    dcl_err = (_col(dc_low_actual_30)  - dc_low_oof["q50"]).dropna()
    dch_err = (_col(dc_high_actual_30) - dc_high_oof["q50"]).dropna()

    # Find dates where ALL FOUR markets have complete 48-period coverage.
    def _complete_days(err: pd.Series) -> set:
        counts = err.groupby(err.index.date).size()
        return set(counts[counts == 48].index)

    common_dates = (
        _complete_days(da_err)
        & _complete_days(bm_err)
        & _complete_days(dcl_err)
        & _complete_days(dch_err)
    )

    valid_dates = sorted(d for d in common_dates if not is_excluded(d))
    n_excluded = len(common_dates) - len(valid_dates)
    logger.info(
        "Error-path pool: %d candidate days, %d excluded "
        "(clock-change / known data-gap dates), %d usable.",
        len(common_dates), n_excluded, len(valid_dates),
    )

    if len(valid_dates) < 30:
        raise ValueError(
            f"Only {len(valid_dates)} usable historical error days found -- "
            "too few to sample a meaningful pool. Check that da_oof/bm_oof/"
            "dc_low_oof/dc_high_oof actually cover the intended CV period."
        )

    def _stack(err: pd.Series) -> np.ndarray:
        rows = []
        for d in valid_dates:
            day_vals = err[err.index.date == d].sort_index().to_numpy()
            rows.append(day_vals)
        return np.stack(rows)

    da_paths  = _stack(da_err)
    bm_paths  = _stack(bm_err)
    dcl_paths = np.stack([_block_avg_1d(row, 48) for row in _stack(dcl_err)])
    dch_paths = np.stack([_block_avg_1d(row, 48) for row in _stack(dch_err)])

    da_vol_7d = _compute_da_vol_7d(da_actual, valid_dates)

    return {
        "dates": np.array(valid_dates),
        "da": da_paths,
        "bm": bm_paths,
        "dc_low": dcl_paths,
        "dc_high": dch_paths,
        "da_vol_7d": da_vol_7d,
    }



# ---------------------------------------------------------------------------
# Step 2: scenario builder using the pool
# ---------------------------------------------------------------------------


def make_historical_error_scenario_builder(
    error_pool: dict,
    n: int = 20,
    da_actual: pd.DataFrame | None = None,
    recency_halflife: float = 90.0,
    error_cap_pct: float = 0.95,
) -> callable:
    """
    Return a scenario_builder that samples n historical joint error days
    (with replacement) and adds each to the target window's own q50 forecast.

    If da_actual is provided, sampling is regime-conditioned and recency-weighted:
      - Volatility similarity: each pool day is weighted by
        exp(-|pool_vol - target_vol| / bandwidth), where pool_vol and target_vol
        are the 7-day trailing DA price std computed before each respective date,
        and bandwidth is the median absolute deviation of pool volatilities.
      - Recency: each pool day is additionally weighted by
        exp(-log(2) / recency_halflife * days_before_target), so a pool day
        recency_halflife days before the target gets half the base weight.

    If da_actual is None, falls back to uniform sampling over all available
    pool dates (original behaviour).
    """
    pool_vols = error_pool.get("da_vol_7d")
    use_regime = da_actual is not None and pool_vols is not None
    if use_regime:
        logger.info(
            "Regime-conditioned sampling enabled (recency half-life=%.0f days). "
            "Bandwidth and caps computed per target date from available pool only.",
            recency_halflife,
        )



    def builder(window_start, horizon, da_fc_win, bm_fc_win, dc_low_fc_win, dc_high_fc_win, seed=None):
        if horizon % 48 != 0:
            raise ValueError(
                f"horizon={horizon} must be a multiple of 48 "
                "(historical error paths are sampled per 48-period day)."
            )
        if horizon % 8 != 0:
            raise ValueError(
                f"horizon={horizon} must be divisible by 8 for DC block averaging."
            )

        rng = np.random.default_rng(seed)
        n_halves = horizon // 48

        target_date = window_start.date()

        efa_groups = _efa_block_groups(window_start, horizon)

        def _block_avg_win(arr):
            result = np.empty_like(arr, dtype=float)
            for blk_start, blk_end in efa_groups:
                result[blk_start:blk_end] = arr[blk_start:blk_end].mean()
            return result

        da_q50  = da_fc_win["q50"].values[:horizon]
        bm_q50  = bm_fc_win["q50"].values[:horizon]
        dcl_q50 = _block_avg_win(dc_low_fc_win["q50"].values[:horizon])
        dch_q50 = _block_avg_win(dc_high_fc_win["q50"].values[:horizon])

        da_scen  = np.zeros((n, horizon))
        bm_scen  = np.zeros((n, horizon))
        dcl_scen = np.zeros((n, horizon))
        dch_scen = np.zeros((n, horizon))



        available = np.where(error_pool["dates"] < target_date)[0]
        if len(available) < n:
            raise ValueError(
                f"Only {len(available)} error-path days available before "
                f"{target_date} (need at least n={n}). Start date is too early "
                "in the error pool's coverage. Consider a later backtest start "
                "date or reduce n."
            )
        
        # Bandwidth from available pool only — no leakage of future vol distribution
        vol_bandwidth = None
        if use_regime:
            avail_vols_all = pool_vols[available]
            valid_avail_vols = avail_vols_all[~np.isnan(avail_vols_all)]
            if len(valid_avail_vols) > 1:
                diffs = np.abs(valid_avail_vols[:, None] - valid_avail_vols[None, :])
                vol_bandwidth = float(np.median(diffs[diffs > 0])) or 10.0
            else:
                vol_bandwidth = 10.0

        # Error caps from available pool only — no leakage of future error distribution
        if 0.0 < error_cap_pct < 1.0:
            da_cap  = float(np.percentile(np.abs(error_pool["da"][available]),      error_cap_pct * 100))
            bm_cap  = float(np.percentile(np.abs(error_pool["bm"][available]),      error_cap_pct * 100))
            dcl_cap = float(np.percentile(np.abs(error_pool["dc_low"][available]),  error_cap_pct * 100))
            dch_cap = float(np.percentile(np.abs(error_pool["dc_high"][available]), error_cap_pct * 100))
        else:
            da_cap = bm_cap = dcl_cap = dch_cap = None

        # Build sampling weights
        sample_weights = None
        if use_regime and vol_bandwidth is not None:
            lookback = pd.date_range(
                end=pd.Timestamp(target_date, tz="UTC") - pd.Timedelta(minutes=30),
                periods=7 * 48, freq="30min",
            )
            target_window = _col(da_actual).reindex(lookback).dropna()
            target_vol = target_window.std() if len(target_window) >= 48 else np.nan

            if not np.isnan(target_vol):
                avail_vols = pool_vols[available]
                vol_ok = ~np.isnan(avail_vols)

                vol_sim = np.where(
                    vol_ok,
                    np.exp(-np.abs(avail_vols - target_vol) / vol_bandwidth),
                    1.0,
                )
                days_ago = np.array(
                    [(target_date - d).days for d in error_pool["dates"][available]]
                )
                recency = np.exp(-np.log(2) / recency_halflife * days_ago)

                w = vol_sim * recency
                w_sum = w.sum()
                if w_sum > 0:
                    raw_weights = w / w_sum

                    # Effective sample size
                    ess = 1.0 / float(np.sum(raw_weights ** 2))

                    # ESS guard: blend with uniform if over-concentrated
                    ess_threshold = float(n)
                    if ess < ess_threshold:
                        alpha = ess / ess_threshold
                        uniform_w = np.ones(len(available)) / len(available)
                        sample_weights = alpha * raw_weights + (1 - alpha) * uniform_w
                        logger.warning(
                            "%s  ESS=%.1f < threshold %.1f — blending regime weights "
                            "with uniform (alpha=%.3f)",
                            target_date, ess, ess_threshold, alpha,
                        )
                    else:
                        sample_weights = raw_weights

                    # Diagnostics
                    avail_dates = error_pool["dates"][available]
                    top5_idx = np.argsort(sample_weights)[-5:][::-1]
                    top5 = [
                        (str(avail_dates[i]), f"{sample_weights[i]:.4f}")
                        for i in top5_idx
                    ]
                    years = np.array([d.year for d in avail_dates])
                    year_shares = {
                        yr: round(float((sample_weights * (years == yr)).sum()), 3)
                        for yr in sorted(set(years))
                    }
                    wtd_avg_vol = float(
                        np.nansum(sample_weights * np.where(vol_ok, avail_vols, np.nan))
                    )
                    logger.info(
                        "%s  target_vol=%.2f  bandwidth=%.2f  ESS=%.1f  "
                        "wtd_pool_vol=%.2f  caps(DA/BM/DCL/DCH)=%.1f/%.1f/%.1f/%.1f",
                        target_date, target_vol, vol_bandwidth, ess, wtd_avg_vol,
                        da_cap or 0, bm_cap or 0, dcl_cap or 0, dch_cap or 0,
                    )
                    logger.info(
                        "%s  top5=%s  year_shares=%s",
                        target_date, top5, year_shares,
                    )

        for half in range(n_halves):
            sl = slice(half * 48, (half + 1) * 48)
            idx = rng.choice(available, size=n, replace=True, p=sample_weights)

            da_err  = np.clip(error_pool["da"][idx],      -da_cap,  da_cap)  if da_cap  is not None else error_pool["da"][idx]
            bm_err  = np.clip(error_pool["bm"][idx],      -bm_cap,  bm_cap)  if bm_cap  is not None else error_pool["bm"][idx]
            dcl_err = np.clip(error_pool["dc_low"][idx],  -dcl_cap, dcl_cap) if dcl_cap is not None else error_pool["dc_low"][idx]
            dch_err = np.clip(error_pool["dc_high"][idx], -dch_cap, dch_cap) if dch_cap is not None else error_pool["dc_high"][idx]

            da_scen[:, sl]  = da_q50[sl][np.newaxis, :]  + da_err
            bm_scen[:, sl]  = bm_q50[sl][np.newaxis, :]  + bm_err
            dcl_scen[:, sl] = dcl_q50[sl][np.newaxis, :] + dcl_err
            dch_scen[:, sl] = dch_q50[sl][np.newaxis, :] + dch_err

        return {
            "da": da_scen,
            "bm": bm_scen,
            "dc_low": dcl_scen,
            "dc_high": dch_scen,
            "seed": seed,
        }

    builder.__name__ = "historical_error_scenario_builder"
    return builder

