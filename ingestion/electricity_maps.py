import logging
from datetime import UTC, datetime

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

ZONE = "GB"


def fetch_electricity_maps() -> None:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:00:00Z")
    url = f"https://api.electricitymaps.com/v3/price-day-ahead/past?zone={ZONE}&datetime={timestamp}&temporalGranularity=hourly"
    response = httpx.get(url, headers={"auth-token": settings.em_api_key}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(
            f"Electricity Maps API returned unexpected shape: expected dict, "
            f"got {type(data).__name__}. Response: {data!r}"
        )
    df = pd.DataFrame([data])
    output_path = settings.data_dir / "raw" / "em_day_ahead" / f"em_day_ahead_{datetime.now(UTC).strftime('%Y%m%d%H')}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Electricity Maps data saved to %s", output_path)
