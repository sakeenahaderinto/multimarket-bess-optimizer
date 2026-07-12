"""
data_quality.py — DA price imputation audit.

Identifies every 30-minute slot where the DA price in the feature pipeline
was forward-filled by more than one step (i.e. genuine imputation, not just
the expected :30 frequency-alignment fill). Classifies each imputed slot by
context and writes two output files:

  data/quality/da_imputed_slots.parquet    — slot-level detail
  data/quality/da_imputed_by_date.parquet  — date-level summary (join key for backtest results)

Context classification
----------------------
Three contexts, ordered by severity:

  (c) settlement  — slot falls in the first SETTLE_PERIODS half-hours of its
                    calendar date; the imputed value is settled against as if
                    it were the real market price. Most severe.
  (b) target      — slot is used directly as the DA training target (the
                    'price' column); imputed labels bias the forecaster.
  (a) lag_feature — slot appears as a lagged input (price_lag_48/96/144/336)
                    for a later row within the study period; imputed features
                    may propagate bias downstream. Least severe.

A slot can belong to multiple contexts simultaneously.

Usage
-----
    python data_quality.py
"""

import logging
from pathlib import Path

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match pipeline.py and backtest/engine.py exactly
# ---------------------------------------------------------------------------
STUDY_START = "2022-01-01"
STUDY_END   = "2025-12-31"
DA_FORWARD_FILL_LIMIT = 96  # must match features/pipeline.py

# Lag periods used in features/lag_features.py (30-min periods)
LAG_PERIODS = [48, 96, 144, 336]

# Settled periods per backtest step (must match backtest/engine.py)
SETTLE_PERIODS = 48


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _compute_fill_steps(raw: pd.Series, canonical_index: pd.DatetimeIndex) -> pd.Series:
    """
    Return how many forward-fill steps were applied to each canonical slot.

    fill_steps == 0  → slot had an observed raw value
    fill_steps == 1  → :30 slot filled from the preceding :00 (expected frequency alignment)
    fill_steps >  1  → slot filled across a genuine data gap (imputation)

    Note: fill_steps are computed without a limit cap so all gaps are
    measured. Slots that fall beyond DA_FORWARD_FILL_LIMIT are NaN in the
    actual feature pipeline and are excluded from the final imputed set in
    the caller.
    """
    raw_on_canonical = raw.reindex(canonical_index)
    is_original = raw_on_canonical.notna()

    # Cumsum-groupby trick: each original slot starts a new group; within
    # each group, cumsum of (~is_original) counts consecutive filled slots.
    group = is_original.cumsum()
    fill_steps = (~is_original).astype(int).groupby(group).cumsum()
    fill_steps[is_original] = 0

    return fill_steps


def run_audit() -> None:
    """Run the DA imputation audit and write both output files."""

    # ----------------------------------------------------------------
    # 1. Load raw DA data
    # ----------------------------------------------------------------
    logger.info("Loading raw DA data...")
    da_dir = settings.data_dir / "raw" / "em_day_ahead"
    da_files = sorted(da_dir.glob("*.parquet"))
    if not da_files:
        raise FileNotFoundError(f"No DA parquet files found in {da_dir}")

    da_raw = pd.concat(
        [pd.read_parquet(f) for f in da_files], ignore_index=True
    )
    da_raw["datetime"] = pd.to_datetime(da_raw["datetime"], utc=True)
    da_raw = da_raw.set_index("datetime").sort_index()["value"]
    da_raw = da_raw[~da_raw.index.duplicated(keep="first")]

    # ----------------------------------------------------------------
    # 2. Build canonical 30-min index (mirrors pipeline.py)
    # ----------------------------------------------------------------
    canonical_index = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(STUDY_END + " 23:30", tz="UTC"),
        freq="30min",
        tz="UTC",
    )
    logger.info(
        "Canonical index: %s → %s (%d slots)",
        canonical_index[0],
        canonical_index[-1],
        len(canonical_index),
    )

    # ----------------------------------------------------------------
    # 3. Compute fill steps for every canonical slot
    # ----------------------------------------------------------------
    logger.info("Computing fill steps...")
    fill_steps = _compute_fill_steps(da_raw, canonical_index)

    # The actual filled series used in pipeline.py (with the cap)
    da_filled = (
        da_raw.resample("30min")
        .ffill(limit=DA_FORWARD_FILL_LIMIT)
        .reindex(canonical_index)
    )

    # Build full audit table
    audit = pd.DataFrame({"fill_steps": fill_steps}, index=canonical_index)
    audit.index.name = "timestamp"
    audit["is_freq_aligned"] = audit["fill_steps"] == 1   # expected :30 alignment
    # Imputed = filled by >1 step AND non-null in the actual pipeline series
    # (slots beyond the fill limit are NaN in pipeline, so they don't affect results)
    audit["is_imputed"] = (audit["fill_steps"] > 1) & da_filled.notna()

    imputed = audit[audit["is_imputed"]].copy()

    n_imputed = len(imputed)
    n_dates   = imputed.index.normalize().nunique()
    logger.info(
        "Imputed slots (fill_steps > 1, within fill limit): %d across %d unique dates.",
        n_imputed,
        n_dates,
    )

    # ----------------------------------------------------------------
    # 4. Context classification
    # ----------------------------------------------------------------

    # (b) Training target: every imputed slot is used directly as the DA price
    #     label for the DA forecaster (pipeline.py sets df["price"] = da_aligned).
    imputed["ctx_training_target"] = True

    # (c) Settlement: slot falls in the first SETTLE_PERIODS half-hours of its
    #     calendar date (midnight–23:30 window settled by the backtest engine).
    slot_of_day = imputed.index.hour * 2 + imputed.index.minute // 30
    imputed["ctx_settlement"] = slot_of_day < SETTLE_PERIODS

    # (a) Lag feature: this slot T is used as a lagged input for row (T + lag_n * 30min).
    #     It qualifies for at least one lag if (T + min_lag_offset) is within study period.
    #     Minimum lag = 48 periods = 24h.  Beyond that, larger lags only extend the range.
    study_end_ts = pd.Timestamp(STUDY_END + " 23:30", tz="UTC")
    min_lag_offset = pd.Timedelta(minutes=30 * min(LAG_PERIODS))  # 24 hours
    imputed["ctx_lag_feature"] = (imputed.index + min_lag_offset) <= study_end_ts

    # ----------------------------------------------------------------
    # 5. Write slot-level output
    # ----------------------------------------------------------------
    out_dir = settings.data_dir / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    slot_path = out_dir / "da_imputed_slots.parquet"
    imputed.to_parquet(slot_path, engine="pyarrow")
    logger.info("Slot-level audit written to %s  (%d rows)", slot_path, len(imputed))

    # ----------------------------------------------------------------
    # 6. Date-level summary — joinable with backtest results on 'date'
    # ----------------------------------------------------------------
    imputed_with_date = imputed.copy()
    imputed_with_date["date"] = pd.DatetimeIndex(
        imputed_with_date.index.normalize()
    )

    date_summary = (
        imputed_with_date.groupby("date")
        .agg(
            n_imputed_total        =("fill_steps",            "count"),
            n_imputed_settlement   =("ctx_settlement",        "sum"),
            n_imputed_target       =("ctx_training_target",   "sum"),
            n_imputed_lag_feature  =("ctx_lag_feature",       "sum"),
            max_fill_steps         =("fill_steps",            "max"),
        )
    )
    date_summary = date_summary.astype({
        "n_imputed_settlement":  int,
        "n_imputed_target":      int,
        "n_imputed_lag_feature": int,
    })
    # Primary join flag: True if any settlement-window slot was imputed on this date
    date_summary["da_imputed"] = date_summary["n_imputed_settlement"] > 0

    date_path = out_dir / "da_imputed_by_date.parquet"
    date_summary.to_parquet(date_path, engine="pyarrow")
    logger.info(
        "Date-level summary written to %s  (%d dates, %d with imputed settlement)",
        date_path,
        len(date_summary),
        int(date_summary["da_imputed"].sum()),
    )

    # Brief console summary
    total_study_days = len(pd.date_range(STUDY_START, STUDY_END, freq="D"))
    logger.info(
        "%.1f%% of study-period dates (%d / %d) have at least one imputed slot "
        "in their settlement window.",
        date_summary["da_imputed"].sum() / total_study_days * 100,
        int(date_summary["da_imputed"].sum()),
        total_study_days,
    )



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    run_audit()
