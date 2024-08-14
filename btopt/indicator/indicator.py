from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..data.timeframe import Timeframe
from ..parameters import Parameters
from ..strategy.helper import Data
from ..types import StrategyType
from ..util.log_config import logger_main
from ..util.metaclasses import PreInitABCMeta


class Indicator(metaclass=PreInitABCMeta):
    def __init__(
        self,
        name: str,
        *args,
        **kwargs,
    ):
        """
        Initialize the Indicator instance.

        Args:
            strategy ('Strategy'): The Strategy object associated with the indicator.
            name (str): The name of the indicator.
            parameters (Optional[Dict[str, Any]]): Additional parameters for the indicator.

        Raises:
            ValueError: If the warmup_period is less than 1.
        """
        self.name: str = name
        self.outputs: List[str] = []

        # Initialize parameters
        self._parameters: Parameters = Parameters(kwargs)
        self._warmup_period: int = 1
        self._current_index: int = 0
        self._is_warmup_complete: bool = False
        self._initialized = False

    def _initialize_indicator(self, **kwargs):
        # Initialize symbols
        self._symbols: List[str] = kwargs.get("symbols", [])

        # Initialize timeframes
        self._timeframes: List[Timeframe] = kwargs.get("timeframes", [])

        # Initialize strategy
        self._strategy: StrategyType = kwargs.get("strategy", None)

        # Initialize strategy outputs. Add a custom column to store SMA values
        for output_name in self.outputs:
            for symbol in self._symbols:
                for timeframe in self._timeframes:
                    self._strategy.datas[symbol].add_custom_column(
                        timeframe, output_name
                    )
                    self.outputs.append(f"{self.name}_{output_name}")

        self._validate_initialization()
        self._initialized = True

    @property
    def datas(self) -> Dict[str, Data]:
        """
        Get the current Data object from the associated Strategy.

        Returns:
            Data: The current Data object.
        """

        return self._strategy.datas

    @property
    def symbols(self):
        return self._symbols

    @property
    def timeframes(self):
        return self._timeframes

    @property
    def parameters(self) -> Parameters:
        """
        Get the indicator parameters.

        Returns:
            Parameters: The indicator parameters object.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters: Union[Parameters, Dict[str, Any]]) -> None:
        """
        Set new indicator parameters.

        Args:
            new_parameters (Dict[str, Any]): A dictionary of new parameter values.
        """
        self._parameters.update()
        logger_main.info(
            f"Updated parameters for indicator {self.name}: {new_parameters}"
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

    def _validate_initialization(self) -> bool:
        """
        Validate the initialization parameters of the Indicator.

        This method checks if:
        1. The indicator has an associated strategy.
        2. The indicator is subscribed to at least one symbol.
        3. All subscribed symbols are valid (present in the strategy's data).
        4. All specified timeframes are valid for each subscribed symbol.

        Returns:
            bool: True if all validations pass.

        Raises:
            ValueError: If any validation fails, with a descriptive error message.
        """
        if not self._strategy or not isinstance(self._strategy, StrategyType):
            logger_main.error(f"Indicator {self.name} has no associated strategy.")
            raise ValueError(f"Indicator {self.name} has no associated strategy.")

        if not self._symbols:
            logger_main.error(
                f"Indicator {self.name} must subscribe to at least one symbol."
            )
            raise ValueError(
                f"Indicator {self.name} must subscribe to at least one symbol."
            )

        invalid_symbols = set(self._symbols) - set(self._strategy.datas.keys())
        if invalid_symbols:
            logger_main.error(
                f"Invalid symbols for indicator {self.name}: {invalid_symbols}"
            )
            raise ValueError(
                f"Invalid symbols for indicator {self.name}: {invalid_symbols}"
            )

        for symbol in self._symbols:
            invalid_timeframes = set(self._timeframes) - set(
                self._strategy.datas[symbol].timeframes
            )
            if invalid_timeframes:
                logger_main.error(
                    f"Invalid timeframes for symbol {symbol} in indicator {self.name}: {invalid_timeframes}"
                )
                raise ValueError(
                    f"Invalid timeframes for symbol {symbol} in indicator {self.name}: {invalid_timeframes}"
                )

        logger_main.info(f"Initialization validation passed for indicator {self.name}")

    @abstractmethod
    def on_data(self) -> None:
        pass

    def _on_data(self) -> None:
        """
        Internal method called when new data is available.

        This method performs the following steps:
        1. Checks if the warmup period is complete.
        2. If warmup is complete, calls the user-defined `on_data` method.
        3. Increments the current index.

        If an exception occurs during the execution of `on_data`, it logs the error
        and re-raises the exception.
        """
        try:
            if self._check_warmup_period():
                if not self._initialized:
                    logger_main.warning(
                        f"indicator {self.name} has not been initialized. Skipping current iteration."
                    )
                    return

                self.on_data()
                self._current_index += 1
        except Exception as e:
            logger_main.log_and_raise(
                f"Error in on_data method of Indicator {self.name}: {str(e)}"
            )

    def _check_warmup_period(self) -> bool:
        """
        Check if all subscribed symbols and timeframes have sufficient data for the warmup period.

        This method verifies that each symbol and timeframe combination has at least
        `warmup_period` number of data points.

        Returns:
            bool: True if all symbol-timeframe combinations have sufficient data, False otherwise.
        """
        if self._is_warmup_complete:
            return True

        for symbol in self._symbols:
            for timeframe in self._timeframes:
                data_length = len(self._strategy.datas[symbol][timeframe])
                if data_length < self._warmup_period:
                    logger_main.debug(
                        f"Insufficient data for Indicator {self.name} on symbol {symbol}, "
                        f"timeframe {timeframe}. Current length: {data_length}, "
                        f"Required: {self._warmup_period}"
                    )
                    return False

        self._is_warmup_complete = True
        logger_main.info(f"Warmup period complete for Indicator {self.name}")
        return True

    def get(
        self,
        output_name: str,
        symbol: Optional[str] = None,
        timeframe: Optional[Union[str, Timeframe]] = None,
        index: int = 0,
    ) -> Any:
        """
        Fetch output/custom column values from the indicator data.

        This method allows flexible retrieval of indicator output values. It can fetch
        values for different symbols, timeframes, and historical indexes.

        Args:
            output_name (str): The name of the output/custom column to fetch.
            symbol (Optional[str]): The symbol to fetch data for. If None, uses the first symbol
                                    in the indicator's symbol list.
            timeframe (Optional[Union[str, Timeframe]]): The timeframe to fetch data for.
                                                         If None, uses the primary timeframe.
            index (int): The historical index to fetch (0 is the most recent). Defaults to 0.

        Returns:
            Any: The value of the specified output/custom column.

        Raises:
            ValueError: If the specified symbol, timeframe, or output_name is invalid.
            IndexError: If the specified index is out of range.
        """
        try:
            # Validate and set symbol
            if symbol is None:
                symbol = self._symbols[0]
            elif symbol not in self._symbols:
                raise ValueError(f"Invalid symbol: {symbol}")

            # Validate and set timeframe
            if timeframe is None:
                timeframe = self._strategy.datas[symbol].primary_timeframe
            elif isinstance(timeframe, str):
                timeframe = Timeframe(timeframe)
            if timeframe not in self._timeframes:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Fetch and return the data
            data = self._strategy.datas[symbol][timeframe][output_name]
            if index >= len(data):
                raise IndexError(
                    f"Index {index} out of range. Available data points: {len(data)}"
                )

            return data[index]

        except Exception as e:
            logger_main.error(f"Error in get method of Indicator {self.name}: {str(e)}")
            raise

    def __repr__(self) -> str:
        """
        Return a string representation of the Indicator.

        Returns:
            str: A string representation of the Indicator.
        """
        return f"{self.name}(warmup_period={self._warmup_period}, parameters={self._parameters})"
