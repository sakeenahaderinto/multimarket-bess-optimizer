import pandas as pd

FORECAST_GATE_OFFSET = pd.Timedelta(hours=13, minutes=30)


def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def create_market_features(
    bm_df: pd.DataFrame,
    da_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    canonical_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Creates cross-market features aligned to a canonical 30-minute index.

    All *_lag_48 features use the fixed gate origin (10:30 D-1) rather than
    a rolling shift(48). Every period on delivery day D gets the same feature
    value — the observation at 10:30 D-1, the last slot before the 11:00 DA
    gate closure.
    """
    _check_required_columns(bm_df, ["netImbalanceVolume", "systemSellPrice"], "bm_df")
    _check_required_columns(da_df, ["value"], "da_df")
    _check_required_columns(weather_df, ["wind_speed_10m"], "weather_df")

    market_df = pd.DataFrame(index=canonical_index)

    # Gate-origin time for each row: 10:30 D-1 from midnight of delivery day D
    gate_times = canonical_index.normalize() - FORECAST_GATE_OFFSET

    # BM imbalance at gate
    bm_imbalance_aligned = bm_df["netImbalanceVolume"].reindex(canonical_index)
    market_df["bm_imbalance_volume_lag_48"] = bm_imbalance_aligned.reindex(gate_times).values

    # Wind change at gate: wind(10:30 D-1) - wind(10:00 D-1)
    wind_aligned = weather_df["wind_speed_10m"].reindex(canonical_index, method="ffill")
    wind_at_gate      = wind_aligned.reindex(gate_times).values
    wind_at_prev      = wind_aligned.reindex(gate_times - pd.Timedelta(minutes=30)).values
    market_df["wind_change_lag_48"] = wind_at_gate - wind_at_prev

    # DA–BM spread at gate
    da_aligned      = da_df["value"].resample("30min").ffill().reindex(canonical_index)
    bm_sell_aligned = bm_df["systemSellPrice"].reindex(canonical_index)
    da_at_gate  = da_aligned.reindex(gate_times).values
    bm_at_gate  = bm_sell_aligned.reindex(gate_times).values
    market_df["spread_lag_48"] = da_at_gate - bm_at_gate

    return market_df




# import pandas as pd


# def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(
#             f"{source}: missing required columns {missing}. "
#             f"Available: {list(df.columns)}"
#         )


# def create_market_features(
#     bm_df: pd.DataFrame,
#     da_df: pd.DataFrame,
#     weather_df: pd.DataFrame,
#     canonical_index: pd.DatetimeIndex,
# ) -> pd.DataFrame:
#     """
#     Creates cross-market features aligned to a canonical 30-minute index.

#     All external series are reindexed onto canonical_index before any shift
#     is applied, so that shift(48) unambiguously means a 24-hour lag regardless
#     of the original frequency of the input series.

#     Timing assumption: all features use a 48-period (24-hour) lag, meaning
#     they only use information available at least 24 hours before the decision
#     time. This is appropriate for day-ahead and balancing market forecasting.

#     Inputs:
#         bm_df:            BM DataFrame with columns ['netImbalanceVolume', 'systemSellPrice'].
#         da_df:            DA DataFrame with column ['value'].
#         weather_df:       Weather DataFrame with column ['wind_speed_10m'].
#         canonical_index:  The shared 30-minute DatetimeIndex for the study period.
#     Outputs:
#         DataFrame with columns: bm_imbalance_volume_lag_48, wind_change_lag_48,
#         spread_lag_48.
#     """
#     _check_required_columns(bm_df, ["netImbalanceVolume", "systemSellPrice"], "bm_df")
#     _check_required_columns(da_df, ["value"], "da_df")
#     _check_required_columns(weather_df, ["wind_speed_10m"], "weather_df")

#     market_df = pd.DataFrame(index=canonical_index)

#     # Align BM series to canonical index first, then lag by 48 periods (24 hours)
#     bm_imbalance_aligned = bm_df["netImbalanceVolume"].reindex(canonical_index)
#     market_df["bm_imbalance_volume_lag_48"] = bm_imbalance_aligned.shift(48)

#     # Align wind speed to canonical index, compute change, then lag by 48 periods (24 hours)
#     # Note: uses realised wind speed, not NWP forecast — see weather_features.py for limitation note
#     wind_aligned = weather_df["wind_speed_10m"].reindex(canonical_index, method="ffill")
#     market_df["wind_change_lag_48"] = wind_aligned.diff().shift(48)

#     # Align DA series to canonical 30-min index before shifting
#     # da_df may be hourly — resample to 30min first to avoid shift(48) != 24h
#     da_aligned = da_df["value"].resample("30min").ffill().reindex(canonical_index)
#     bm_sell_aligned = bm_df["systemSellPrice"].reindex(canonical_index)
#     market_df["spread_lag_48"] = da_aligned.shift(48) - bm_sell_aligned.shift(48)  # DA–BM spread, 24-hour lag

#     return market_df