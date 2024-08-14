import numpy as np

from ..indicator import Indicator
from ..util.log_config import logger_main


class SimpleMovingAverage(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Simple Moving Average (SMA) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the SimpleMovingAverage indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the SimpleMovingAverage indicator."
            )

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["sma"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the SMA period."""
        return self._period

    @property
    def source(self) -> int:
        """Get the SMA calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Simple Moving Average for each timeframe when new data arrives.
        """

        for symbol in self.symbols:
            self.outputs[symbol].sma = self._current_index

            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            sma_value = self._calculate_sma(price_data)
            self.outputs[symbol]["sma"] = sma_value

    def _calculate_sma(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Simple Moving Average for the given data.

        Args:
            data (np.ndarray): The price data to calculate the SMA for.

        Returns:
            float: The SMA value.
        """
        sma = np.sum(data) / self._period
        return sma

    def __repr__(self) -> str:
        """
        Return a string representation of the SimpleMovingAverage indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"SimpleMovingAverage(name={self.name}, period={self._period}, price_type={self._price_type}, symbol={self._symbol})"


SMA = SimpleMovingAverage
