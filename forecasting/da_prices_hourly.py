"""
da_prices_hourly.py — Phase 4: hourly-resolution DA forecaster.

Mirrors DAPricesForecaster's feature_cols exactly (BASE_FEATURES plus
wind_change_lag_48/spread_lag_48, here at their hourly-equivalent _lag_24
naming) so any difference in calibration is attributable to resolution
alone, not a different feature set.

Overrides load_features() since BaseForecaster.load_features() hardcodes
features.parquet -- this points at a separate file instead, so training
this model does not touch or depend on the main 30-min feature table that
BM, DC Low, DC High, and the original 30-min DAPricesForecaster all use.

model_id = "da_hourly" ensures save_oof_forecasts/save_oos_forecasts write
to da_hourly_oof_*.parquet / da_hourly_oos_*.parquet -- separate files,
no collision with the existing da_oof_*/da_oos_* used elsewhere.
"""

import pandas as pd

from config import settings
from .base import BaseForecaster

DA_HOURLY_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "year_sin",
    "year_cos",
    "demand_lag_24",
    "demand_lag_48",
    "demand_lag_72",
    "demand_lag_168",
    "price_lag_24",
    "price_lag_48",
    "price_lag_72",
    "price_lag_168",
    "demand_roll_mean_4",
    "demand_roll_std_4",
    "price_roll_mean_4",
    "price_roll_std_4",
    "demand_roll_mean_24",
    "demand_roll_std_24",
    "price_roll_mean_24",
    "price_roll_std_24",
    "demand_roll_mean_168",
    "demand_roll_std_168",
    "price_roll_mean_168",
    "price_roll_std_168",
    "wind_speed",
    "cloud_cover",
    "bm_imbalance_volume_lag_24",
    "wind_change_lag_24",
    "spread_lag_24",
]


class DAPricesForecasterHourly(BaseForecaster):
    model_id = "da_hourly"
    target_col = "price"
    feature_cols = DA_HOURLY_FEATURES

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(settings.data_dir / "features" / "features_da_hourly.parquet")