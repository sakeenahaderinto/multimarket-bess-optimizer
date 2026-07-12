import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from forecasting.bm_da_spread import BMDASpreadForecaster

BMDASpreadForecaster().run()
