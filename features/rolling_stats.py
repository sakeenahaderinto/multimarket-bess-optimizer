import pandas as pd


def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean and standard deviation features for demand and price.

    Each series is shifted by 48 periods (24 hours) before the rolling window
    is applied, so the window only uses information at least 24 hours old.
    This prevents any look-ahead leakage.

    Rolling windows and their calendar equivalents at 30-min resolution:
        Window 8   =  4 hours  (8  x 30 min)
        Window 48  = 24 hours  (48 x 30 min)
        Window 336 =  7 days   (336 x 30 min)

    Inputs:
        df: DataFrame with a 30-minute DatetimeIndex and columns
            ['demand', 'price'].
    Outputs:
        df with rolling mean and std columns added for each window.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    _check_required_columns(df, ["demand", "price"], "add_rolling_features")

    # Window sizes and their calendar equivalents at 30-min resolution
    rolling_windows = [
        (8,   "4 hours"),
        (48,  "24 hours / 1 day"),
        (336, "7 days / 1 week"),
    ]

    for window, _ in rolling_windows:
        # shift(48) applied before rolling: window only uses data ≥24 hours old
        df[f"demand_roll_mean_{window}"] = df["demand"].shift(48).rolling(window=window, min_periods=window//2).mean()
        df[f"demand_roll_std_{window}"]  = df["demand"].shift(48).rolling(window=window, min_periods=window//2).std()
        df[f"price_roll_mean_{window}"]  = df["price"].shift(48).rolling(window=window, min_periods=window//2).mean()
        df[f"price_roll_std_{window}"]   = df["price"].shift(48).rolling(window=window, min_periods=window//2).std()

    return df