import logging

import pandas as pd

from config import settings
from .cyclical_time import add_cyclical_time
from .lag_features import add_lag_features
from .rolling_stats import add_rolling_features
from .market_features import create_market_features
from .weather_features import add_weather_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Study period
# These bounds define the canonical 30-minute index for all feature and
# target construction. Setting them explicitly (rather than deriving from
# whichever raw data source happens to start/end earliest) makes the
# experiment window reproducible and independent of data ingestion coverage.
# ---------------------------------------------------------------------------
STUDY_START = "2022-01-01"  # inclusive
STUDY_END   = "2025-12-31"  # inclusive (last 30-min slot = 2025-12-31 23:30)
DA_FORWARD_FILL_LIMIT = 96  # allows DA values to be forward-filled up to 48 hours (96 x 30 min)
                            # fills beyond one half-hour step are recorded by data_quality.py


def build_features() -> None:
    """
    Orchestrates the full feature engineering pipeline.

    Steps:
        1. Load raw data from parquet sources.
        2. Establish a canonical 30-minute DatetimeIndex for the study period (STUDY_START-STUDY_END).
        3. Align all input series to that index.
        4. Build target columns (DA price, BM price, DC Low, DC High).
        5. Call each feature module in turn.
        6. Write the final feature table to parquet.

    """

    # ------------------------------------------------------------------ 
    # 1. Load raw data                                                     
    # ------------------------------------------------------------------ 
    bm_df = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da_df = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")
    weather_df = pd.concat(
        [pd.read_parquet(f) for f in sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))],
        ignore_index=True,
    )
    dc_df = pd.read_parquet(settings.data_dir / "processed" / "dc_auction_processed.parquet")

    # ------------------------------------------------------------------ 
    # 2. Parse and sort indices                                            
    # ------------------------------------------------------------------ 
    da_df["datetime"] = pd.to_datetime(da_df["datetime"], utc=True)
    da_df = da_df.set_index("datetime").sort_index()

    bm_df["startTime"] = pd.to_datetime(bm_df["startTime"], utc=True)
    bm_df = bm_df.set_index("startTime").sort_index()

    weather_df = weather_df.set_index("date").sort_index()
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)
    weather_df = weather_df[~weather_df.index.duplicated(keep="last")]

    dc_df["delivery_start"] = pd.to_datetime(dc_df["delivery_start"], utc=True)
    dc_df = dc_df.set_index("delivery_start").sort_index()

    # ------------------------------------------------------------------ 
    # 3. Build canonical 30-minute index and align all series             
    # ------------------------------------------------------------------ 
    canonical_index = pd.date_range(
        start=pd.Timestamp(STUDY_START, tz="UTC"),
        end=pd.Timestamp(STUDY_END + " 23:30", tz="UTC"),
        freq="30min",
        tz="UTC",
    )

    # Warn if any input series has >5% missing values after alignment
    def _check_coverage(series: pd.Series, name: str, threshold: float = 0.05) -> None:
        missing_rate = series.isna().mean()
        if missing_rate > threshold:
            logger.warning(
                "%s has %.1f%% missing values after alignment to the canonical index.",
                name,
                missing_rate * 100,
            )

    da_aligned     = da_df["value"].resample("30min").ffill(limit=DA_FORWARD_FILL_LIMIT).reindex(canonical_index)
    bm_price_aligned  = bm_df["systemSellPrice"].reindex(canonical_index)
    bm_demand_aligned = bm_df["totalAcceptedOfferVolume"].reindex(canonical_index)

    dcl_aligned = (
        dc_df[dc_df["service"] == "DCL"]["clearing_price"]
        .resample("30min").ffill(limit=7)
        .reindex(canonical_index)
    )
    dch_aligned = (
        dc_df[dc_df["service"] == "DCH"]["clearing_price"]
        .resample("30min").ffill(limit=7)
        .reindex(canonical_index)
    )

    _check_coverage(da_aligned,      "DA price")
    _check_coverage(bm_price_aligned, "BM system sell price")
    _check_coverage(dcl_aligned,     "DC Low clearing price")
    _check_coverage(dch_aligned,     "DC High clearing price")

    # ------------------------------------------------------------------
    # 4. Build the base feature DataFrame from the canonical index      
    # ------------------------------------------------------------------
    df = pd.DataFrame(index=canonical_index)

    df["price"]  = da_aligned   # forward-fill DA gaps up to 48 hours
    df["demand"] = bm_demand_aligned

    # Target columns — must exist before lag_features is called
    df["bm_price"]      = bm_price_aligned
    df["dc_low_price"]  = dcl_aligned
    df["dc_high_price"] = dch_aligned

    # ------------------------------------------------------------------
    # 5. Call feature modules                                           
    # ------------------------------------------------------------------
    df = add_cyclical_time(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_features(df, weather_df)

    market_df = create_market_features(bm_df, da_df, weather_df, canonical_index)
    df = pd.concat([df, market_df], axis=1)

    # ------------------------------------------------------------------
    # 6. Write final feature table                                       
    # ------------------------------------------------------------------
    output_path = settings.data_dir / "features" / "features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")

    logger.info("Feature table written to %s — shape: %s", output_path, df.shape)