import pandas as pd
import numpy as np

def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclical time features to the DataFrame.

    Uses fractional hour (hour + minute/60) so that half-hourly settlement
    periods are correctly distinguished — e.g. 10:00 -> 10.0, 10:30 -> 10.5.

    Inputs:
        df: DataFrame with a 30-minute DatetimeIndex.
    Outputs:
        df with added columns: hour_sin, hour_cos, dayofweek_sin,
        dayofweek_cos, year_sin, year_cos.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Use fractional hour so 10:00 and 10:30 are distinct settlement periods
    fractional_hour = df.index.hour + df.index.minute / 60

    df["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)

    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # 365.2425 accounts for leap years in the yearly cycle
    df["year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.2425)
    df["year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.2425)

    return df