import logging
from datetime import UTC, datetime

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_bmrs() -> None:
    response = httpx.get(
        f"{BASE_URL}/balancing/settlement/system-prices/{datetime.now(UTC).strftime('%Y-%m-%d')}?format=json",
        timeout=30,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json()["data"]).convert_dtypes()
    output_path = settings.data_dir / "raw" / "bmrs" / f"disebsp_{datetime.now(UTC).strftime('%Y%m%d%H')}.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("BMRS data saved to %s", output_path)
