from .base import BASE_FEATURES, BaseForecaster


class BMPriceForecaster(BaseForecaster):
    model_id = "bm"
    target_col = "bm_price"
    feature_cols = BASE_FEATURES + ["wind_change_lag_48", "spread_lag_48"]
