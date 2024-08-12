from abc import abstractmethod
from typing import Any, Dict, Optional

from ..log_config import logger_main
from ..parameters import Parameters
from ..strategy.helper import Data
from ..util.metaclasses import PreInitABCMeta


class Indicator(metaclass=PreInitABCMeta):
    def __init__(
        self,
        data: Data,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Indicator instance.

        Args:
            data (Data): The Data object associated with the indicator.
            name (str): The name of the indicator.
            warmup_period (int): The number of bars required before the indicator produces valid output.
            **kwargs: Additional parameters for the indicator.

        Raises:
            ValueError: If the warmup_period is less than 1.
        """
        self.name: str = name
        self.data: Data = data
        self._parameters: Parameters = Parameters(parameters or {})
        self._warmup_period: int = 1
        self._current_index: int = 0
        self._is_warmup_complete: bool = False

        logger_main.info(f"Initialized {self.name} indicator")

    @property
    def parameters(self) -> Parameters:
        """
        Get the strategy parameters.

        Returns:
            Parameters: The strategy parameters object.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Set new strategy parameters.

        Args:
            new_parameters (Dict[str, Any]): A dictionary of new parameter values.
        """
        self._parameters = Parameters(new_parameters)
        logger_main.info(
            f"Updated parameters for strategy {self.name}: {new_parameters}"
        )

    @property
    def warmup_period(self) -> int:
        """
        Get the warmup period for the indicator.

        Returns:
            int: The current warmup period.
        """
        return self._warmup_period

    @warmup_period.setter
    def warmup_period(self, value: int) -> None:
        """
        Set the warmup period for the indicator.

        Args:
            value (int): The new warmup period.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            logger_main.log_and_raise(
                ValueError("warmup_period must be a positive integer.")
            )
        self._warmup_period = value
        logger_main.info(f"Updated warmup_period to {value}")

    @abstractmethod
    def on_data(self) -> None:
        """
        Process new data.

        This method is called for each new bar of data. It should update the indicator's state
        and call the calculate method to update the outputs.
        """
        pass

    def _on_data(self) -> None:
        """
        Perform the indicator calculation.

        This method should implement the specific logic for calculating the indicator's outputs
        based on the current state and historical data.
        """

        if self._check_warmup_period():
            self.on_data()

    def _check_warmup_period(self) -> bool:
        """
        Check if all timeframes within self.dats have at least `warmup_period` data points.

        Returns:
            bool: True if all timeframes have sufficient data, False otherwise.
        """
        if self._is_warmup_complete:
            return True

        timeframes = self.data.timeframes

        for timeframe in timeframes:
            if len(self.data[timeframe]) < self.warmup_period:
                logger_main.warning(
                    f"Insufficient data for Indicator `{self.name}` on `{timeframe}` timeframe."
                )
                return False

        self._is_warmup_complete = True
        logger_main.warning(f"Indicator {self._id} warm up is complete.")
        return True

    def get(self, name: str) -> Optional[Any]:
        """
        Get the current value of a specific output.

        Args:
            name (str): The name of the output to retrieve.

        Returns:
            Optional[Any]: The current value of the specified output, or None if the warmup period is not complete.

        Raises:
            AttributeError: If the specified output does not exist.
        """
        if not self._is_warmup_complete:
            logger_main.info(
                f"{self.name} indicator warmup period not complete, returning None"
            )
            return None

        try:
            return self.outputs.get(name)
        except AttributeError:
            logger_main.error(
                f"Output '{name}' does not exist for {self.name} indicator"
            )
            raise

    def __repr__(self) -> str:
        """
        Return a string representation of the Indicator.

        Returns:
            str: A string representation of the Indicator.
        """
        return f"{self.name}(warmup_period={self._warmup_period}, parameters={self._parameters})"
