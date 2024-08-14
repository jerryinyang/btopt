import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class WeightedMovingAverage(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Weighted Moving Average (WMA) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the WeightedMovingAverage indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the WeightedMovingAverage indicator."
            )

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["wma"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the WMA period."""
        return self._period

    @property
    def source(self) -> str:
        """Get the WMA calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Weighted Moving Average for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            wma_value = self._calculate_wma(price_data)
            self.outputs[symbol]["wma"] = wma_value

    def _calculate_wma(self, data: np.ndarray) -> float:
        """
        Calculate the Weighted Moving Average for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the WMA for.

        Returns:
            float: The WMA value.
        """
        wma = talib.WMA(self.float_reverse(data), timeperiod=self._period)
        return wma[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the WeightedMovingAverage indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"WeightedMovingAverage(name={self.name}, period={self._period}, source={self._source})"


WMA = WeightedMovingAverage
