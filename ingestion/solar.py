import logging
from datetime import UTC, datetime

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pvlive.uk/pvlive/api/v4"
HEADERS = {"Accept-Encoding": "gzip, deflate"}


def fetch_solar() -> None:
    params = {
        "start": (datetime.now(UTC) - pd.Timedelta(hours=1)).isoformat(),
        "end": datetime.now(UTC).isoformat(),
    }
    response = httpx.get(f"{BASE_URL}/gsp/0", headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["data"], columns=data["meta"])
    output_path = settings.data_dir / "raw" / "solar" / f"solar_{datetime.now(UTC).strftime('%Y%m%d%H')}.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Solar data saved to %s", output_path)
