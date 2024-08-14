import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class ExponentialMovingAverage(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Exponential Moving Average (EMA) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the ExponentialMovingAverage indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the ExponentialMovingAverage indicator."
            )

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["ema"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the EMA period."""
        return self._period

    @property
    def source(self) -> str:
        """Get the EMA calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Exponential Moving Average for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            ema_value = self._calculate_ema(price_data)
            self.outputs[symbol]["ema"] = ema_value

    def _calculate_ema(self, data: np.ndarray) -> float:
        """
        Calculate the Exponential Moving Average for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the EMA for.

        Returns:
            float: The EMA value.
        """
        ema = talib.EMA(self.float_reverse(data), timeperiod=self._period)
        return ema[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the ExponentialMovingAverage indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"ExponentialMovingAverage(name={self.name}, period={self._period}, source={self._source})"


EMA = ExponentialMovingAverage
