import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class BollingerBands(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Bollinger Bands indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the BollingerBands indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the BollingerBands indicator."
            )
        if "deviation" not in self.parameters:
            logger_main.log_and_raise(
                "`deviation` parameter is required for the BollingerBands indicator."
            )

        self._period = self.parameters.get("period", 20)
        self._source = self.parameters.get("source", "close")
        self._deviation = self.parameters.get("deviation", 2)

        self.output_names = ["upper", "middle", "lower"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the Bollinger Bands period."""
        return self._period

    @property
    def source(self) -> str:
        """Get the Bollinger Bands calculation source."""
        return self._source

    @property
    def deviation(self) -> float:
        """Get the Bollinger Bands standard deviation multiplier."""
        return self._deviation

    def on_data(self) -> None:
        """
        Calculate the Bollinger Bands for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            upper, middle, lower = self._calculate_bollinger_bands(price_data)
            self.outputs[symbol]["upper"] = upper
            self.outputs[symbol]["middle"] = middle
            self.outputs[symbol]["lower"] = lower

    def _calculate_bollinger_bands(self, data: np.ndarray) -> tuple:
        """
        Calculate the Bollinger Bands for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the Bollinger Bands for.

        Returns:
            tuple: The upper, middle, and lower Bollinger Bands values.
        """
        upper, middle, lower = talib.BBANDS(
            self.float_reverse(data),
            timeperiod=self._period,
            nbdevup=self._deviation,
            nbdevdn=self._deviation,
            matype=talib.MA_Type.SMA,
        )
        return upper[-1], middle[-1], lower[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the BollingerBands indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"BollingerBands(name={self.name}, period={self._period}, source={self._source}, deviation={self._deviation})"


BBANDS = BollingerBands
