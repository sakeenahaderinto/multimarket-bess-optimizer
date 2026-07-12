from .base import BASE_FEATURES, BaseForecaster


class DCLowForecaster(BaseForecaster):
    model_id = "dc_low"
    target_col = "dc_low_price"
    feature_cols = BASE_FEATURES


class DCHighForecaster(BaseForecaster):
    model_id = "dc_high"
    target_col = "dc_high_price"
    feature_cols = BASE_FEATURES
