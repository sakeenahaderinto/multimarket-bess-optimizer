import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from forecasting.da_prices import DAPricesForecaster
from forecasting.bm_price import BMPriceForecaster
from forecasting.dc_tender import DCLowForecaster, DCHighForecaster
from features.pipeline import build_features



build_features()


DAPricesForecaster().run()
BMPriceForecaster().run()
DCLowForecaster().run()
DCHighForecaster().run()

