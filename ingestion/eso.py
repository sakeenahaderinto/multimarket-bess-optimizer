import logging
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


def fetch_eso() -> None:
    seven_days_ago = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    sql = (
        f'SELECT * FROM "596f29ac-0387-4ba4-a6d3-95c243140707" '
        f'WHERE "deliveryStart" >= \'{seven_days_ago}\' ORDER BY "deliveryStart" DESC'
    )
    response = httpx.get(
        "https://api.neso.energy/api/3/action/datastore_search_sql",
        params={"sql": sql},
        timeout=60,
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json()["result"]["records"])
    output_path = settings.data_dir / "raw" / "eso" / f"eso_{datetime.now(UTC).strftime('%Y%m%d%H')}.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("ESO data saved to %s", output_path)
