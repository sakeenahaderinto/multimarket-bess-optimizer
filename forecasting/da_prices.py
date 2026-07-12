from .base import BASE_FEATURES, BaseForecaster


class DAPricesForecaster(BaseForecaster):
    model_id = "da"
    target_col = "price"
    feature_cols = BASE_FEATURES + ["wind_change_lag_48", "spread_lag_48"]
