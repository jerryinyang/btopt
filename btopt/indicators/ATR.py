import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class AverageTrueRange(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Average True Range (ATR) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the AverageTrueRange indicator."
            )

        self._period = self.parameters.get("period", 14)

        self.output_names = ["atr"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the ATR period."""
        return self._period

    def on_data(self) -> None:
        """
        Calculate the Average True Range for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the high, low, and close data for the symbol and timeframe
            high_data = self.datas[symbol].get(
                self.timeframe,
                column="high",
                size=self._period,
            )
            low_data = self.datas[symbol].get(
                self.timeframe,
                column="low",
                size=self._period,
            )
            close_data = self.datas[symbol].get(
                self.timeframe,
                column="close",
                size=self._period,
            )
            atr_value = self._calculate_atr(high_data, low_data, close_data)
            self.outputs[symbol]["atr"] = atr_value

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """
        Calculate the Average True Range for the given data using Ta-lib.

        Args:
            high (np.ndarray): The high price data.
            low (np.ndarray): The low price data.
            close (np.ndarray): The close price data.

        Returns:
            float: The ATR value.
        """
        atr = talib.ATR(
            self.float_reverse(high),
            self.float_reverse(low),
            self.float_reverse(close),
            timeperiod=self._period,
        )
        return atr[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the AverageTrueRange indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"AverageTrueRange(name={self.name}, period={self._period})"


ATR = AverageTrueRange
