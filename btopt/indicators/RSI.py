import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class RelativeStrengthIndex(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Relative Strength Index (RSI) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the RelativeStrengthIndex indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the RelativeStrengthIndex indicator."
            )

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["rsi"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the RSI period."""
        return self._period

    @property
    def source(self) -> str:
        """Get the RSI calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Relative Strength Index for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            rsi_value = self._calculate_rsi(price_data)
            self.outputs[symbol]["rsi"] = rsi_value

    def _calculate_rsi(self, data: np.ndarray) -> float:
        """
        Calculate the Relative Strength Index for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the RSI for.

        Returns:
            float: The RSI value.
        """
        rsi = talib.RSI(self.float_reverse(data), timeperiod=self._period)
        return rsi[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the RelativeStrengthIndex indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"RelativeStrengthIndex(name={self.name}, period={self._period}, source={self._source})"


RSI = RelativeStrengthIndex
