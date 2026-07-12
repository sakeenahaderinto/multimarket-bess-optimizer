import pandas as pd


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

    Lag periods assume a 30-minute DatetimeIndex:
        lag 48  = 24 hours  (1 day)
        lag 96  = 48 hours  (2 days)
        lag 144 = 72 hours  (3 days)
        lag 336 = 168 hours (1 week)

    If the index is not at 30-minute frequency, these lags will not correspond
    to the intended calendar durations. Always align all series to a canonical
    30-minute index in pipeline.py before calling this function.

    Inputs:
        df: DataFrame with a 30-minute DatetimeIndex and at minimum
            columns ['demand', 'price'].
    Outputs:
        df with lag columns added for demand, price, and optionally
        bm_price, dc_low_price, dc_high_price.
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

    # Lag periods and their calendar equivalents at 30-min resolution
    lag_periods = [
        (48,  "24 hours / 1 day"),
        (96,  "48 hours / 2 days"),
        (144, "72 hours / 3 days"),
        (336, "168 hours / 1 week"),
    ]

    for lag, _ in lag_periods:
        df[f"demand_lag_{lag}"] = df["demand"].shift(lag)
        df[f"price_lag_{lag}"] = df["price"].shift(lag)

        if "bm_price" in df.columns:
            df[f"bm_price_lag_{lag}"] = df["bm_price"].shift(lag)
        if "dc_low_price" in df.columns:
            df[f"dc_low_price_lag_{lag}"] = df["dc_low_price"].shift(lag)
        if "dc_high_price" in df.columns:
            df[f"dc_high_price_lag_{lag}"] = df["dc_high_price"].shift(lag)

    return df