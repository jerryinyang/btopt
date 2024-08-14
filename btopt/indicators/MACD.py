import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class MovingAverageConvergenceDivergence(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Moving Average Convergence Divergence (MACD) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "fast_period" not in self.parameters:
            logger_main.log_and_raise(
                "`fast_period` parameter is required for the MovingAverageConvergenceDivergence indicator."
            )
        if "slow_period" not in self.parameters:
            logger_main.log_and_raise(
                "`slow_period` parameter is required for the MovingAverageConvergenceDivergence indicator."
            )
        if "signal_period" not in self.parameters:
            logger_main.log_and_raise(
                "`signal_period` parameter is required for the MovingAverageConvergenceDivergence indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the MovingAverageConvergenceDivergence indicator."
            )

        self._fast_period = self.parameters.get("fast_period", 12)
        self._slow_period = self.parameters.get("slow_period", 26)
        self._signal_period = self.parameters.get("signal_period", 9)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["macd", "signal", "histogram"]
        self.warmup_period = max(
            self._fast_period, self._slow_period, self._signal_period
        )

    @property
    def fast_period(self) -> int:
        """Get the MACD fast period."""
        return self._fast_period

    @property
    def slow_period(self) -> int:
        """Get the MACD slow period."""
        return self._slow_period

    @property
    def signal_period(self) -> int:
        """Get the MACD signal period."""
        return self._signal_period

    @property
    def source(self) -> str:
        """Get the MACD calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Moving Average Convergence Divergence for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self.warmup_period,
            )
            macd, signal, histogram = self._calculate_macd(price_data)
            self.outputs[symbol]["macd"] = macd
            self.outputs[symbol]["signal"] = signal
            self.outputs[symbol]["histogram"] = histogram

    def _calculate_macd(self, data: np.ndarray) -> tuple:
        """
        Calculate the Moving Average Convergence Divergence for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the MACD for.

        Returns:
            tuple: The MACD line, signal line, and histogram values.
        """
        macd, signal, histogram = talib.MACD(
            self.float_reverse(data),
            fastperiod=self._fast_period,
            slowperiod=self._slow_period,
            signalperiod=self._signal_period,
        )
        return macd[-1], signal[-1], histogram[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the MovingAverageConvergenceDivergence indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"MovingAverageConvergenceDivergence(name={self.name}, fast_period={self._fast_period}, slow_period={self._slow_period}, signal_period={self._signal_period}, source={self._source})"


MACD = MovingAverageConvergenceDivergence
