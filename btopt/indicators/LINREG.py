import numpy as np
import talib

from ..indicator import Indicator
from ..util.log_config import logger_main


class LinearRegression(Indicator):
    def __init__(
        self,
        name: str,
        **kwargs,
    ):
        """
        Initialize the Linear Regression indicator.

        Args:
            name (str): The name of this indicator instance.
        """
        super().__init__(name, **kwargs)

        if "period" not in self.parameters:
            logger_main.log_and_raise(
                "`period` parameter is required for the LinearRegression indicator."
            )
        if "source" not in self.parameters:
            logger_main.log_and_raise(
                "`source` parameter is required for the LinearRegression indicator."
            )

        self._period = self.parameters.get("period", 14)
        self._source = self.parameters.get("source", "close")

        self.output_names = ["linear_reg", "slope", "intercept", "angle"]
        self.warmup_period = self._period

    @property
    def period(self) -> int:
        """Get the Linear Regression period."""
        return self._period

    @property
    def source(self) -> str:
        """Get the Linear Regression calculation source."""
        return self._source

    def on_data(self) -> None:
        """
        Calculate the Linear Regression for each symbol when new data arrives.
        """
        for symbol in self.symbols:
            # Fetch the data for the symbol and timeframe
            price_data = self.datas[symbol].get(
                self.timeframe,
                column=self._source,
                size=self._period,
            )
            linear_reg, slope, intercept, angle = self._calculate_linear_regression(
                price_data
            )
            self.outputs[symbol]["linear_reg"] = linear_reg
            self.outputs[symbol]["slope"] = slope
            self.outputs[symbol]["intercept"] = intercept
            self.outputs[symbol]["angle"] = angle

    def _calculate_linear_regression(self, data: np.ndarray) -> tuple:
        """
        Calculate the Linear Regression for the given data using Ta-lib.

        Args:
            data (np.ndarray): The price data to calculate the Linear Regression for.

        Returns:
            tuple: The Linear Regression line, slope, intercept, and angle values.
        """
        data = self.float_reverse(data)
        linear_reg = talib.LINEARREG(data, timeperiod=self._period)
        slope = talib.LINEARREG_SLOPE(data, timeperiod=self._period)
        intercept = talib.LINEARREG_INTERCEPT(data, timeperiod=self._period)
        angle = talib.LINEARREG_ANGLE(data, timeperiod=self._period)

        return linear_reg[-1], slope[-1], intercept[-1], angle[-1]

    def __repr__(self) -> str:
        """
        Return a string representation of the LinearRegression indicator.

        Returns:
            str: A string representation of the indicator.
        """
        return f"LinearRegression(name={self.name}, period={self._period}, source={self._source})"


LINREG = LinearRegression
