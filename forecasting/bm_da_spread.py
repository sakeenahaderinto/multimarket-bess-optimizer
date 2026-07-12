import pandas as pd

from .base import BASE_FEATURES, BaseForecaster


class BMDASpreadForecaster(BaseForecaster):
    model_id = "spread"
    target_col = "spread"
    feature_cols = BASE_FEATURES + [
        "wind_change_lag_48",
        "bm_price_lag_48",
        "bm_price_lag_96",
        "bm_price_lag_144",
        "bm_price_lag_336",
    ]

    def load_features(self) -> pd.DataFrame:
        df = super().load_features()
        df["spread"] = df["bm_price"] - df["price"]
        return df
