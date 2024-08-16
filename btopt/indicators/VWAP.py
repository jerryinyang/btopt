from datetime import datetime
from typing import Dict

import numpy as np

from ..indicator import Indicator
from ..util.log_config import logger_main


class VolumeWeightedAveragePrice(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Volume Weighted Average Price (VWAP) indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        self.output_names = ["vwap"]
        self.warmup_period = 1  # VWAP can be calculated from the first data point

        # Initialize cumulative values for each symbol
        self._cumulative_tp: Dict[str, float] = {}
        self._cumulative_volume: Dict[str, float] = {}
        self._last_reset: Dict[str, datetime] = {}

    def on_data(self) -> None:
        """
        Calculate the VWAP for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Get the current bar data
            bar = self.datas[symbol].get(self.timeframe)
            current_time = bar.timestamp

            # Check if we need to reset cumulative values (new trading day)
            if self._should_reset(symbol, current_time):
                self._reset_cumulative_values(symbol)

            # Calculate typical price
            typical_price = (bar.high + bar.low + bar.close) / 3

            # Update cumulative values
            self._cumulative_tp[symbol] += typical_price * bar.volume
            self._cumulative_volume[symbol] += bar.volume

            # Calculate VWAP
            vwap = (
                self._cumulative_tp[symbol] / self._cumulative_volume[symbol]
                if self._cumulative_volume[symbol] > 0
                else np.nan
            )

            # Store the VWAP value
            self.outputs[symbol]["vwap"] = vwap

    def _should_reset(self, symbol: str, current_time: datetime) -> bool:
        """
        Check if we should reset the cumulative values for a new trading day.

        Args:
            symbol (str): The symbol to check.
            current_time (datetime): The current bar's timestamp.

        Returns:
            bool: True if we should reset, False otherwise.
        """
        if symbol not in self._last_reset:
            return True

        last_reset = self._last_reset[symbol]
        return last_reset.date() < current_time.date()

    def _reset_cumulative_values(self, symbol: str) -> None:
        """
        Reset the cumulative values for a symbol at the start of a new trading day.

        Args:
            symbol (str): The symbol to reset values for.
        """
        self._cumulative_tp[symbol] = 0.0
        self._cumulative_volume[symbol] = 0.0
        self._last_reset[symbol] = self.datas[symbol].get_current_timestamp(
            self.timeframe
        )
        logger_main.info(f"Reset cumulative values for {symbol} VWAP calculation")

    def __repr__(self) -> str:
        """
        Return a string representation of the VolumeWeightedAveragePrice indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"VolumeWeightedAveragePrice(name={self.name})"


VWAP = VolumeWeightedAveragePrice
