import pandas as pd


def _check_required_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged weather features to the DataFrame.

    All weather variables are shifted by 48 periods (24 hours) to avoid
    using future realised weather observations as features.

    SIMPLIFICATION NOTE: In a real pre-market deployment, Numerical Weather
    Prediction (NWP) forecasts — not lagged actuals — would be used here,
    since forecasts are what is actually available before gate closure.
    Lagged realised weather is used as a simplification because archived
    NWP forecast data is not available for this study. This should be stated
    explicitly as a limitation in any published work.

    Inputs:
        df:          Main feature DataFrame with a 30-minute DatetimeIndex.
        weather_df:  Weather DataFrame with columns ['temperature_2m',
                     'wind_speed_10m', 'cloud_cover', 'shortwave_radiation'].
    Outputs:
        df with added columns: temperature, wind_speed, cloud_cover, irradiance.
        All features carry a 48-period (24-hour) lag.
    """
    _check_required_columns(
        weather_df,
        ["temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"],
        "weather_df",
    )

    # Reindex to main DataFrame index, forward-fill gaps, then lag by 48 periods (24 hours)
    df["temperature"] = weather_df["temperature_2m"].reindex(df.index, method="ffill").shift(48)
    df["wind_speed"]  = weather_df["wind_speed_10m"].reindex(df.index, method="ffill").shift(48)
    df["cloud_cover"] = weather_df["cloud_cover"].reindex(df.index, method="ffill").shift(48)
    df["irradiance"]  = weather_df["shortwave_radiation"].reindex(df.index, method="ffill").shift(48)

    return df