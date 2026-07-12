"""
excluded_dates.py — Dates excluded from the historical error-path sampling
pool (Phase 2), and from being treated as normal-structure target days.

A date is excluded if its settlement-period structure deviates from the
standard 48-period day, for any reason: a UK clock-change transition, a
genuine data gap (BM's six anomaly days), or a source-boundary seam. 
Sampling a jointDA/BM/DC error path *from* one of these dates, or adding 
a sampled error path *onto* one of these dates as a target, would inject 
a structural artifact (misaligned period count, or a boundary discontinuity) 
rather than a genuine forecast error -- exactly the kind of fabricated "false
peak or trough" warned, just for a data-structure reason instead of a market reason.

All DC fiscal-year boundaries (pre2023->FY2023, FY2023->FY2024,
FY2024->FY2025) were confirmed clean (zero gap, zero overlap, checked
per-service) -- per findings.md Section 1.5 and the 2025-06-26 follow-up
confirming the two boundaries not checked in that session are also clean.
None of the DC source-boundary dates need to be excluded.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# UK clock-change dates, 2021-2025 (last Sunday of March / October).
# Spring: 46 settlement periods. Autumn: 50 settlement periods.
# Source: confirmed against BM and DA completeness checks;
# the same calendar dates apply to all 30-min-or-finer
# series (BM, DA once resampled, DC once resampled).
# ---------------------------------------------------------------------------
CLOCK_CHANGE_DATES = pd.to_datetime([
    "2021-03-28", "2021-10-31",
    "2022-03-27", "2022-10-30",
    "2023-03-26", "2023-10-29",
    "2024-03-31", "2024-10-27",
    "2025-03-30", "2025-10-26",
]).date

# ---------------------------------------------------------------------------
# BM genuine data-gap anomaly days -- not clock changes, real missing
# settlement periods. Source: findings.md Section 1.1.
# ---------------------------------------------------------------------------
BM_ANOMALY_DATES = pd.to_datetime([
    "2021-01-07",  # 47 periods, missing 1
    "2022-05-31",  # 46 periods, missing 2
    "2022-10-22",  # 46 periods, missing 2
    "2023-01-17",  # 47 periods, missing 1
    "2023-01-22",  # 45 periods, missing 3
    "2023-03-18",  # 46 periods, missing 2
]).date

# ---------------------------------------------------------------------------
# Combined exclusion set
# ---------------------------------------------------------------------------
EXCLUDED_DATES = set(CLOCK_CHANGE_DATES) | set(BM_ANOMALY_DATES)


def is_excluded(date) -> bool:
    """
    True if `date` (anything pd.Timestamp(date).date() can parse) has a
    non-standard settlement-period structure and should not be used as
    either a source (sampled-from) or target (sampled-onto) day for
    historical error-path scenario generation.
    """
    return pd.Timestamp(date).date() in EXCLUDED_DATES


def filter_valid_dates(dates) -> pd.DatetimeIndex:
    """Return only the dates in `dates` that are NOT excluded."""
    dates = pd.DatetimeIndex(dates)
    mask = ~pd.Series(dates.date, index=dates).isin(EXCLUDED_DATES)
    return dates[mask.values]