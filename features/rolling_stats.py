import pandas as pd

from features.lag_features import FORECAST_GATE_OFFSET


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

    All rolling statistics are pinned to the gate-origin timestamp (10:30 D-1)
    for each delivery day D. The rolling window is computed on the full raw
    series first (giving a value at every 30-min timestamp), then reindexed
    to 10:30 D-1. That single value is assigned to all 48 periods of day D.

    This ensures no period of day D can see data from after gate closure,
    regardless of where it falls within the delivery day. The previous
    shift(48) approach was clean only for periods 0-21; periods 22-47 used
    data up to 13 hours after gate closure.

    Rolling windows and their calendar equivalents at 30-min resolution:
        Window 8   =  4 hours  (8  x 30 min)
        Window 48  = 24 hours  (48 x 30 min)
        Window 336 =  7 days   (336 x 30 min)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    _check_required_columns(df, ["demand", "price"], "add_rolling_features")

    gate_times = df.index.normalize() - FORECAST_GATE_OFFSET

    for window in [8, 48, 336]:
        for col in ["demand", "price"]:
            roll_mean = df[col].rolling(window=window, min_periods=window // 2).mean()
            roll_std  = df[col].rolling(window=window, min_periods=window // 2).std()
            df[f"{col}_roll_mean_{window}"] = roll_mean.reindex(gate_times).values
            df[f"{col}_roll_std_{window}"]  = roll_std.reindex(gate_times).values

    return df