"""
pipeline_da_hourly.py — Standalone hourly-resolution feature
pipeline for the DA forecaster.

Built separately from features/pipeline.py (continues to serve BM, DC Low, DC High,
and the original 30-min DA model used for comparison). 
DA's true underlying signal only changes once per hour -- the existing 30-min 
pipeline trains on a ffilled, duplicated-looking
target, and the trained model has been shown to produce spuriously
different q10/q50/q90 forecasts between the :00 and :30 of the same hour.

This module builds an entirely separate canonical HOURLY index and feature
table for DA, matching base.py's BASE_FEATURES exactly (so any difference
in model performance is attributable to resolution alone, not a smaller
feature set):

  - hour/dayofweek/year cyclical features (simplified -- no fractional_hour,
    since minute is always 0 at hourly resolution)
  - price_lag_{24,48,72,168}, price_roll_mean/std_{4,24,168}      (DA price)
  - demand_lag_{24,48,72,168}, demand_roll_mean/std_{4,24,168}    (BM demand)
  - wind_speed, cloud_cover                                       (weather)
  - bm_imbalance_volume_lag_24                                    (BM, summed across each hour's 
                                                                  two real 30-min values -- see 
                                                                  add_bm_imbalance_hourly)

NOTE on bm_imbalance_volume: unlike DA price, BM imbalance volume is
genuinely 30-min-resolution data (not an artifact of ffilling an hourly
source). Converting it to hourly therefore means aggregating two real,
independent half-hour values, not just removing a duplicate -- summed here
(not averaged), since volume is a quantity, not a rate.

Output is later upsampled back to 30-min by exact duplication (not ffill)
before being handed to anything that expects the 30-min grid -- see
upsample_da_forecast() at the bottom of this module.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Study period -- same bounds as the main pipeline, for consistency
# ---------------------------------------------------------------------------
STUDY_START = "2022-01-01"
STUDY_END   = "2025-12-31"
DA_FORWARD_FILL_LIMIT_HOURS = 48  # was 96 half-hours (48h) in the 30-min pipeline


# ---------------------------------------------------------------------------
# Cyclical time features -- simplified for hourly resolution
# ---------------------------------------------------------------------------


def add_cyclical_time_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclical time features for an hourly-indexed DataFrame.

    Unlike cyclical_time.py's add_cyclical_time (which uses fractional_hour
    = hour + minute/60 to distinguish 30-min settlement periods within an
    hour), this uses plain integer hour-of-day, since there is no within-hour
    distinction to make at hourly resolution -- minute is always 0. This is
    the structural fix for the spurious sub-hourly variance found in the
    30-min DA model: there is no feature here that could let the model learn
    a within-hour difference, because there are no within-hour rows.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    hour = df.index.hour

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df["year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.2425)
    df["year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.2425)

    return df


# ---------------------------------------------------------------------------
# Lag and rolling features -- generic, halved periods, hourly index
# (covers both 'price' and 'demand', matching BASE_FEATURES's pattern of
# building the same lag/rolling set for both series)
# ---------------------------------------------------------------------------


def add_lag_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged price and demand features at hourly resolution.

    Lag periods assume an hourly DatetimeIndex:
        lag 24  = 24 hours  (1 day)   -- was lag 48  at 30-min
        lag 48  = 48 hours  (2 days)  -- was lag 96  at 30-min
        lag 72  = 72 hours  (3 days)  -- was lag 144 at 30-min
        lag 168 = 168 hours (1 week)  -- was lag 336 at 30-min

    Inputs:
        df: DataFrame with an hourly DatetimeIndex and columns ['demand', 'price'].
    Outputs:
        df with lag columns added for demand and price.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq not in ("h", "H", "1h", "1H"):
        raise ValueError(
            f"add_lag_features_hourly expects an hourly DatetimeIndex. "
            f"Got: '{inferred_freq}'. Align to hourly resolution first."
        )

    missing = [c for c in ("demand", "price") if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_lag_features_hourly: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    for lag in (24, 48, 72, 168):
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
        df[f"price_lag_{lag}"]  = df["price"].shift(lag)

    return df


def add_rolling_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean/std features for demand and price at hourly resolution.

    Each series is shifted by 24 periods (24 hours) before the rolling
    window is applied, matching the 30-min pipeline's no-look-ahead
    guarantee (shift(48) at 30-min == shift(24) at hourly == 24 hours).

    Rolling windows and their calendar equivalents at hourly resolution:
        Window 4   =  4 hours  -- was window 8   at 30-min
        Window 24  = 24 hours  -- was window 48  at 30-min
        Window 168 =  7 days   -- was window 336 at 30-min

    Inputs:
        df: DataFrame with an hourly DatetimeIndex and columns ['demand', 'price'].
    Outputs:
        df with rolling mean/std columns added for each window.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    missing = [c for c in ("demand", "price") if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_rolling_features_hourly: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    for window in (4, 24, 168):
        for col in ("demand", "price"):
            shifted = df[col].shift(24)
            df[f"{col}_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=window // 2).mean()
            df[f"{col}_roll_std_{window}"]  = shifted.rolling(window=window, min_periods=window // 2).std()

    return df


# ---------------------------------------------------------------------------
# Weather features -- hourly resolution
# ---------------------------------------------------------------------------


def add_weather_features_hourly(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged weather features at hourly resolution.

    Weather is forward-filled to the canonical hourly index then shifted by
    24 periods (24 hours), matching the 30-min pipeline's 48-period (24h)
    lag and ffill treatment in weather_features.py. Weather's own native
    resolution does not need downsampling the way BM imbalance volume does
    -- it was already being ffilled across multiple 30-min slots in the
    original pipeline, so ffilling onto an hourly index is equally valid
    and loses no real information either way.

    Inputs:
        df:          Main hourly feature DataFrame with an hourly DatetimeIndex.
        weather_df:  Weather DataFrame with columns ['wind_speed_10m',
                     'cloud_cover'] at minimum (temperature and irradiance are not used here).
    Outputs:
        df with added columns: wind_speed, cloud_cover. Both carry a
        24-period (24-hour) lag.
    """
    missing = [c for c in ("wind_speed_10m", "cloud_cover") if c not in weather_df.columns]
    if missing:
        raise ValueError(
            f"add_weather_features_hourly: missing required columns {missing} "
            f"in weather_df. Available: {list(weather_df.columns)}"
        )

    df["wind_speed"]  = weather_df["wind_speed_10m"].reindex(df.index, method="ffill").shift(24)
    df["cloud_cover"] = weather_df["cloud_cover"].reindex(df.index, method="ffill").shift(24)

    return df


# ---------------------------------------------------------------------------
# BM imbalance volume -- hourly resolution (downsampled from real 30-min data)
# ---------------------------------------------------------------------------


def add_bm_imbalance_hourly(df: pd.DataFrame, bm_imbalance_30min: pd.Series) -> pd.DataFrame:
    """
    Adds the lagged BM imbalance volume feature at hourly resolution.

    Unlike DA price, BM imbalance volume is genuinely 30-min-resolution
    data -- both half-hours within an hour are real, independent
    observations, not a duplicate-by-construction artifact. Converting to
    hourly therefore means summing the two real half-hour volumes (it is a
    quantity, not a rate), not just collapsing a duplicate.

    Lag is 24 hours (lag 24 at hourly == lag 48 at 30-min, matching
    bm_imbalance_volume_lag_48 in BASE_FEATURES).

    Inputs:
        df:                  Main hourly feature DataFrame with an hourly
                              DatetimeIndex.
        bm_imbalance_30min:  netImbalanceVolume series at native 30-min
                              resolution (e.g. market_features.py's
                              bm_imbalance_aligned, before any lag is applied).
    Outputs:
        df with column bm_imbalance_volume_lag_24 added.
    """
    hourly_sum = bm_imbalance_30min.resample("1h").sum()
    df["bm_imbalance_volume_lag_24"] = hourly_sum.reindex(df.index).shift(24)
    return df


# ---------------------------------------------------------------------------
# DA-specific market features -- hourly equivalents of wind_change_lag_48
# and spread_lag_48 from market_features.py, needed to fully match
# DAPricesForecaster.feature_cols (BASE_FEATURES + these two extras)
# ---------------------------------------------------------------------------


def add_market_features_hourly(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    bm_sell_price_30min: pd.Series,
) -> pd.DataFrame:
    """
    Adds wind_change_lag_24 and spread_lag_24 -- hourly equivalents of
    market_features.py's wind_change_lag_48 and spread_lag_48.

    wind_change_lag_24: wind speed ffilled to hourly, then .diff() (one
    real hour-to-hour change, not a half-hour-to-half-hour change that
    would partly reflect the same ffilled value), then shifted 24 hours.

    spread_lag_24: DA price minus BM system sell price, both at hourly
    resolution, shifted 24 hours. BM sell price is a rate-like quantity
    (£/MWh), so it is averaged across each hour's two real half-hour
    values, not summed.

    Inputs:
        df:                   Main hourly feature DataFrame, must already
                              have 'price' (DA) populated.
        weather_df:           Raw weather DataFrame with wind_speed_10m.
        bm_sell_price_30min:  systemSellPrice at native 30-min resolution
                              (pipeline.py's bm_price_aligned, pre-lag).
    Outputs:
        df with wind_change_lag_24 and spread_lag_24 added.
    """
    if "price" not in df.columns:
        raise ValueError("add_market_features_hourly requires 'price' to already be set on df.")

    wind_hourly = weather_df["wind_speed_10m"].reindex(df.index, method="ffill")
    df["wind_change_lag_24"] = wind_hourly.diff().shift(24)

    bm_sell_hourly = bm_sell_price_30min.resample("1h").mean().reindex(df.index)
    df["spread_lag_24"] = df["price"].shift(24) - bm_sell_hourly.shift(24)

    return df


# ---------------------------------------------------------------------------
# Build the full hourly DA feature table
# ---------------------------------------------------------------------------


def build_da_hourly_features(
    da_df_hourly: pd.DataFrame,
    bm_demand_30min: pd.Series,
    bm_imbalance_30min: pd.Series,
    bm_sell_price_30min: pd.Series,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete hourly DA feature table, matching
    DAPricesForecaster.feature_cols exactly: BASE_FEATURES plus
    wind_change_lag_48/spread_lag_48 (here, _lag_24 at hourly resolution).

    Inputs:
        da_df_hourly:         Output of process_day_ahead_data(), with an
                              hourly DatetimeIndex and a 'value' column --
                              DA's true native resolution, before any
                              30-min resampling/ffill.
        bm_demand_30min:      totalAcceptedOfferVolume at native 30-min
                              resolution (pipeline.py's bm_demand_aligned,
                              pre-lag).
        bm_imbalance_30min:   netImbalanceVolume at native 30-min resolution
                              (market_features.py's bm_imbalance_aligned,
                              pre-lag).
        bm_sell_price_30min:  systemSellPrice at native 30-min resolution
                              (pipeline.py's bm_price_aligned, pre-lag).
        weather_df:           Raw weather DataFrame with wind_speed_10m and
                              cloud_cover columns.

    Outputs:
        Feature DataFrame on a canonical hourly index spanning
        STUDY_START-STUDY_END, with target column 'price' and all features
        matching DAPricesForecaster.feature_cols's scope.
    """
    canonical_index = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(STUDY_END + " 23:00", tz="UTC"),
        freq="h",
        tz="UTC",
    )

    da_aligned = (
        da_df_hourly["value"]
        .resample("1h").ffill(limit=DA_FORWARD_FILL_LIMIT_HOURS)
        .reindex(canonical_index)
    )
    # totalAcceptedOfferVolume is a volume/energy quantity (MWh per period),
    # not a rate -- sum the two real half-hour values, same treatment as
    # bm_imbalance_volume in add_bm_imbalance_hourly.
    demand_aligned = bm_demand_30min.resample("1h").sum().reindex(canonical_index)

    missing_rate = da_aligned.isna().mean()
    if missing_rate > 0.05:
        logger.warning(
            "Hourly DA price has %.1f%% missing values after alignment to "
            "the canonical hourly index.", missing_rate * 100,
        )

    df = pd.DataFrame(index=canonical_index)
    df["price"] = da_aligned
    df["demand"] = demand_aligned

    df = add_cyclical_time_hourly(df)
    df = add_lag_features_hourly(df)
    df = add_rolling_features_hourly(df)
    df = add_weather_features_hourly(df, weather_df)
    df = add_bm_imbalance_hourly(df, bm_imbalance_30min)
    df = add_market_features_hourly(df, weather_df, bm_sell_price_30min)

    return df


# ---------------------------------------------------------------------------
# Upsample hourly forecasts back to 30-min for downstream use
# ---------------------------------------------------------------------------


def upsample_da_forecast(hourly_fc: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate each hour's forecast (q10/q50/q90) across both half-hour
    slots, producing an exactly-duplicated 30-min series -- not ffilled,
    not independently predicted. This guarantees :00 and :30 of the same
    hour are identical by construction, fixing the artifact found in the
    original 30-min-trained DA model.

    Inputs:
        hourly_fc: DataFrame with an hourly DatetimeIndex, columns
                   including q10/q50/q90 (or whatever forecast columns
                   the model produces).
    Outputs:
        DataFrame on a 30-min DatetimeIndex covering the same span, with
        every pair of half-hour rows identical within each hour.
    """
    if not isinstance(hourly_fc.index, pd.DatetimeIndex):
        raise ValueError("hourly_fc must have a DatetimeIndex.")

    thirty_min_index = pd.date_range(
        start=hourly_fc.index.min(),
        end=hourly_fc.index.max() + pd.Timedelta(minutes=30),
        freq="30min",
        tz=hourly_fc.index.tz,
    )

    upsampled = hourly_fc.reindex(thirty_min_index.floor("h")).copy()
    upsampled.index = thirty_min_index

    return upsampled