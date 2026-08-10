import pandas as pd

from features.lag_features import FORECAST_GATE_OFFSET


def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds weather features to the DataFrame, pinned to the gate-origin (10:30 D-1).

    Weather is first reindexed to the main 30-min index (forward-filling any
    gaps), then the value at 10:30 D-1 is assigned to all 48 periods of
    delivery day D. This matches the gate-origin treatment used for all other
    restricted features and ensures no period of day D sees weather observations
    from after gate closure.

    The previous shift(48) approach was clean only for periods 0-21; periods
    22-47 used observations from up to 13 hours after gate closure.

    SIMPLIFICATION NOTE: In a real pre-market deployment, Numerical Weather
    Prediction (NWP) forecasts — not lagged actuals — would be used here,
    since forecasts are what is actually available before gate closure.
    Lagged realised weather is used as a simplification because archived
    NWP forecast data is not available for this study.

    Inputs:
        df:          Main feature DataFrame with a 30-minute DatetimeIndex.
        weather_df:  Weather DataFrame with columns ['temperature_2m',
                     'wind_speed_10m', 'cloud_cover', 'shortwave_radiation'].
    Outputs:
        df with added columns: temperature, wind_speed, cloud_cover, irradiance.
    """
    _check_required_columns(
        weather_df,
        ["temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"],
        "weather_df",
    )

    gate_times = df.index.normalize() - FORECAST_GATE_OFFSET

    # Align weather to the main 30-min index (ffill gaps), then pin to gate-origin
    weather_aligned = weather_df.reindex(df.index, method="ffill")

    df["temperature"] = weather_aligned["temperature_2m"].reindex(gate_times).values
    df["wind_speed"]  = weather_aligned["wind_speed_10m"].reindex(gate_times).values
    df["cloud_cover"] = weather_aligned["cloud_cover"].reindex(gate_times).values
    df["irradiance"]  = weather_aligned["shortwave_radiation"].reindex(gate_times).values

    return df