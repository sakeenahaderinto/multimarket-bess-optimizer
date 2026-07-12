"""
train_da_hourly.py — Phase 4: build hourly DA features and train
DAPricesForecasterHourly in one run.

Combines what would otherwise be two separate steps (build_da_hourly_features.py
+ a training call) into a single script, mirroring train_forecasters.py's
shape for the existing 30-min models.

Usage:
    uv run train_da_hourly.py
"""

import logging

import pandas as pd

from config import settings
from features.pipeline_da_hourly import build_da_hourly_features
from forecasting.da_prices_hourly import DAPricesForecasterHourly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_and_save_features() -> None:
    """
    Loads raw data at native resolution and builds the hourly DA feature
    table, writing it to the path DAPricesForecasterHourly.load_features()
    expects. Mirrors features/pipeline.py's steps 1-2, stopping short of
    the canonical-30min reindex since the hourly pipeline needs native
    resolution instead.
    """
    logger.info("Loading raw data...")
    bm_df = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da_df = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")
    weather_df = pd.concat(
        [pd.read_parquet(f) for f in sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))],
        ignore_index=True,
    )

    da_df["datetime"] = pd.to_datetime(da_df["datetime"], utc=True)
    da_df_hourly = da_df.set_index("datetime").sort_index()  # DA's true native (hourly) resolution

    bm_df["startTime"] = pd.to_datetime(bm_df["startTime"], utc=True)
    bm_df = bm_df.set_index("startTime").sort_index()

    weather_df = weather_df.set_index("date").sort_index()
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)

    bm_sell_price_30min = bm_df["systemSellPrice"]
    bm_demand_30min      = bm_df["totalAcceptedOfferVolume"]
    bm_imbalance_30min   = bm_df["netImbalanceVolume"]

    logger.info("Building hourly DA feature table...")
    df = build_da_hourly_features(
        da_df_hourly, bm_demand_30min, bm_imbalance_30min, bm_sell_price_30min, weather_df
    )

    output_path = settings.data_dir / "features" / "features_da_hourly.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")

    logger.info("Hourly DA feature table written to %s — shape: %s", output_path, df.shape)


def main() -> None:
    build_and_save_features()

    logger.info("Starting training for DAPricesForecasterHourly...")
    DAPricesForecasterHourly().run()


if __name__ == "__main__":
    main()