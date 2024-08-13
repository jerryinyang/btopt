import numpy as np

from ..util.log_config import logger_main
from .indicator import Indicator


class SimpleMovingAverage(Indicator):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Initialize the Simple Moving Average (SMA) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, *args, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise("`period` parameter is required.")
        if "source" not in self.parameters:
            logger_main.log_and_raise("`source` parameter is required.")

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")
        self.warmup_period = self._period

        logger_main.info(
            f"Initialized {self.name} indicator with period {self._period}"
        )

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
            for timeframe in self.datas[self._symbol].timeframes:
                price_data = self.datas[self._symbol][timeframe][self._price_type]
                sma_values = self._calculate_sma(price_data)

                # Update the custom column with the calculated SMA values
                self.datas[self._symbol][timeframe][f"{self.name}_sma"] = sma_values

    def _calculate_sma(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the Simple Moving Average for the given data.

        Args:
            data (np.ndarray): The price data to calculate the SMA for.

        Returns:
            np.ndarray: An array of SMA values.
        """
        sma = np.convolve(data, np.ones(self._period), "valid") / self._period
        return np.concatenate((np.full(self._period - 1, np.nan), sma))

    def __repr__(self) -> str:
        """
        Return a string representation of the SimpleMovingAverage indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"SimpleMovingAverage(name={self.name}, period={self._period}, price_type={self._price_type}, symbol={self._symbol})"
