import pandas as pd

FORECAST_GATE_OFFSET = pd.Timedelta(hours=13, minutes=30)
# Gate = 10:30 D-1 from midnight of delivery day D.
# For each period on delivery day D: normalize() gives midnight D (UTC),
# then subtract 13h30m -> 10:30 D-1.


def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged price and demand features to the DataFrame.

    All *_lag_48 columns use a fixed gate origin (10:30 D-1) rather than
    a rolling shift(48). Every settlement period on delivery day D gets the
    same value — the observation at 10:30 D-1, the last slot before the
    11:00 DA gate closure. This prevents post-gate observations from leaking
    into features for later periods of the same delivery day.

    Lags of 96, 144, 336 periods (48h, 72h, 1 week) are always safely before
    the D-1 gate and remain as rolling shifts.

    Lag periods at 30-minute resolution:
        lag_48  = gate origin (10:30 D-1)  -- fixed per delivery day
        lag_96  = 48 hours  (2 days)
        lag_144 = 72 hours  (3 days)
        lag_336 = 168 hours (1 week)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq not in ("30min", "30T"):
        raise ValueError(
            f"add_lag_features expects a 30-minute DatetimeIndex. "
            f"Got: '{inferred_freq}'. Align all series to 30min in pipeline.py before calling."
        )

    _check_required_columns(df, ["demand", "price"], "add_lag_features")

    # Gate-origin lookup: same value for all 48 periods of delivery day D
    gate_times = df.index.normalize() - FORECAST_GATE_OFFSET  # 10:30 D-1 for each row

    gate_cols = ["demand", "price"]
    for col in ["bm_price", "dc_low_price", "dc_high_price"]:
        if col in df.columns:
            gate_cols.append(col)

    for col in gate_cols:
        df[f"{col}_lag_48"] = df[col].reindex(gate_times).values

    # Safe rolling lags (always 48h+ before any period on D — no gate violation)
    rolling_cols = ["demand", "price"]
    for col in ["bm_price", "dc_low_price", "dc_high_price"]:
        if col in df.columns:
            rolling_cols.append(col)

    for lag in [96, 144, 336]:
        for col in rolling_cols:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


# import pandas as pd


# def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(
#             f"{source}: missing required columns {missing}. "
#             f"Available: {list(df.columns)}"
#         )


# def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds lagged price and demand features to the DataFrame.

#     Lag periods assume a 30-minute DatetimeIndex:
#         lag 48  = 24 hours  (1 day)
#         lag 96  = 48 hours  (2 days)
#         lag 144 = 72 hours  (3 days)
#         lag 336 = 168 hours (1 week)

#     If the index is not at 30-minute frequency, these lags will not correspond
#     to the intended calendar durations. Always align all series to a canonical
#     30-minute index in pipeline.py before calling this function.

#     Inputs:
#         df: DataFrame with a 30-minute DatetimeIndex and at minimum
#             columns ['demand', 'price'].
#     Outputs:
#         df with lag columns added for demand, price, and optionally
#         bm_price, dc_low_price, dc_high_price.
#     """
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("The DataFrame index must be a DatetimeIndex.")

#     inferred_freq = pd.infer_freq(df.index)
#     if inferred_freq not in ("30min", "30T"):
#         raise ValueError(
#             f"add_lag_features expects a 30-minute DatetimeIndex. "
#             f"Got: '{inferred_freq}'. Align all series to 30min in pipeline.py before calling."
#         )

#     _check_required_columns(df, ["demand", "price"], "add_lag_features")

#     # Lag periods and their calendar equivalents at 30-min resolution
#     lag_periods = [
#         (48,  "24 hours / 1 day"),
#         (96,  "48 hours / 2 days"),
#         (144, "72 hours / 3 days"),
#         (336, "168 hours / 1 week"),
#     ]

#     for lag, _ in lag_periods:
#         df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
#         df[f"price_lag_{lag}"] = df["price"].shift(lag)

#         if "bm_price" in df.columns:
#             df[f"bm_price_lag_{lag}"] = df["bm_price"].shift(lag)
#         if "dc_low_price" in df.columns:
#             df[f"dc_low_price_lag_{lag}"] = df["dc_low_price"].shift(lag)
#         if "dc_high_price" in df.columns:
#             df[f"dc_high_price_lag_{lag}"] = df["dc_high_price"].shift(lag)

#     return df