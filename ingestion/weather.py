import logging
from datetime import UTC, datetime

import openmeteo_requests
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

_client = openmeteo_requests.Client()

URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]
PARAMS = {
    "latitude": 51.51,
    "longitude": -0.13,
    "hourly": HOURLY_VARS,
    "timezone": "Europe/London",
    "past_days": 7,
    "forecast_days": 3,
}


def fetch_weather() -> None:
    responses = _client.weather_api(URL, params=PARAMS)
    hourly = responses[0].Hourly()
    data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var_name in enumerate(HOURLY_VARS):
        data[var_name] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    output_path = settings.data_dir / "raw" / "weather" / f"weather_{datetime.now(UTC).strftime('%Y%m%d%H')}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Weather data saved to %s", output_path)
