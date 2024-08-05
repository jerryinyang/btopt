from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main
from ..order import Order
from .helper import BarManager, Data


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    This class provides a framework for creating specific trading strategies.
    It defines the interface and common functionality that all strategy
    implementations should follow.

    Attributes:
        name (str): The name of the strategy.
        parameters (Dict[str, Any]): Dictionary of strategy parameters.
        primary_timeframe (Optional[Timeframe]): The primary timeframe for the strategy.
        datas (Dict[str, Dict[Timeframe, Data]]): Nested dictionary to store data for each symbol and timeframe.
        bar_manager (BarManager): Manager for storing and accessing Bar objects.
        _initialized (bool): Flag indicating whether the strategy has been initialized.
        _primary_symbol (Optional[str]): The primary symbol for the strategy.
    """

    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        primary_timeframe: Optional[Timeframe] = None,
    ) -> None:
        """
        Initialize the Strategy instance.

        Args:
            name (str): The name of the strategy.
            parameters (Optional[Dict[str, Any]], optional): Initial strategy parameters. Defaults to None.
            primary_timeframe (Optional[Timeframe], optional): The primary timeframe for the strategy. Defaults to None.
        """
        self.name: str = name
        self.parameters: Dict[str, Any] = parameters or {}
        self.primary_timeframe: Optional[Timeframe] = primary_timeframe
        self.datas: Dict[str, Dict[Timeframe, Data]] = defaultdict(dict)
        self.bar_manager: BarManager = BarManager()
        self._initialized: bool = False
        self._primary_symbol: Optional[str] = None

    def initialize(
        self,
        symbols: List[str],
        timeframes: List[Timeframe],
        default_timeframe: Timeframe,
    ) -> None:
        """
        Initialize the strategy with symbols and timeframes.

        Args:
            symbols (List[str]): List of symbols the strategy will trade.
            timeframes (List[Timeframe]): List of timeframes the strategy will use.
            default_timeframe (Timeframe): Default timeframe to use if primary_timeframe is not set.
        """
        if self.primary_timeframe is None:
            self.primary_timeframe = default_timeframe
        elif self.primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self.primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self.primary_timeframe = default_timeframe

        for symbol in symbols:
            for timeframe in timeframes:
                self.datas[symbol][timeframe] = Data(symbol, timeframe)

        self._primary_symbol = symbols[0] if symbols else None
        self._initialized = True
        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self.primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

    @property
    def data(self) -> Data:
        """
        Get the primary data stream for the strategy.

        Returns:
            Data: The primary Data object for the strategy.

        Raises:
            ValueError: If the strategy hasn't been initialized or if there's no primary symbol.
            KeyError: If the primary symbol or timeframe is not found in the data dictionary.
        """
        if not self._initialized:
            raise ValueError("Strategy has not been initialized.")
        if self._primary_symbol is None:
            raise ValueError("No primary symbol set for the strategy.")
        try:
            return self.datas[self._primary_symbol][self.primary_timeframe]
        except KeyError:
            raise KeyError(
                f"Data for primary symbol {self._primary_symbol} and timeframe {self.primary_timeframe} not found."
            )

    @abstractmethod
    def generate_signals(self, bar: Bar) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on the current market data.

        This method must be implemented by concrete strategy classes.

        Args:
            bar (Bar): The latest price bar data.

        Returns:
            List[Dict[str, Any]]: A list of signal dictionaries.
        """
        pass

    def on_bar(self, bar: Bar) -> None:
        """
        Handle the arrival of a new price bar.

        This method updates the data structures with the new bar and can be
        overridden by concrete strategy classes to implement custom logic.

        Args:
            bar (Bar): The new price bar data.
        """
        symbol, timeframe = bar.ticker, bar.timeframe
        self.datas[symbol][timeframe].add_bar(bar)
        self.bar_manager.add_bar(bar)

    def __getitem__(self, key: Union[str, Tuple[str, Timeframe]]) -> Data:
        """
        Allow easy access to data streams.

        Args:
            key (Union[str, Tuple[str, Timeframe]]):
                If str, assumes primary timeframe and returns data for that symbol.
                If tuple, returns data for the specified (symbol, timeframe) pair.

        Returns:
            Data: The requested Data object.

        Raises:
            KeyError: If the requested symbol or timeframe is not found.
        """
        if isinstance(key, str):
            return self.datas[key][self.primary_timeframe]
        elif isinstance(key, tuple) and len(key) == 2:
            symbol, timeframe = key
            return self.datas[symbol][timeframe]
        else:
            raise KeyError(f"Invalid key: {key}")

    def get_data(self, symbol: str, timeframe: Optional[Timeframe] = None) -> Data:
        """
        Retrieve a specific data stream.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Optional[Timeframe]): The timeframe to retrieve data for.
                                             If None, uses the primary timeframe.

        Returns:
            Data: The requested Data object.

        Raises:
            KeyError: If the requested symbol or timeframe is not found.
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
        return self.datas[symbol][timeframe]

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set or update the strategy parameters.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameters to set or update.
        """
        self.parameters.update(parameters)
        logger_main.info(f"Updated parameters for strategy {self.name}: {parameters}")

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current strategy parameters.

        Returns:
            Dict[str, Any]: The current strategy parameters.
        """
        return self.parameters

    def validate_parameters(self) -> bool:
        """
        Validate the strategy parameters.

        This method should be implemented by concrete strategy classes to
        perform strategy-specific parameter validation.

        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        return True

    def create_order(
        self,
        symbol: str,
        direction: Order.Direction,
        size: float,
        order_type: Order.ExecType,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a standardized order dictionary.

        Args:
            symbol (str): The symbol to trade.
            direction (Order.Direction): The direction of the trade (LONG or SHORT).
            size (float): The size of the order.
            order_type (Order.ExecType): The type of order (e.g., MARKET, LIMIT).
            price (Optional[float]): The price for limit orders.
            **kwargs: Additional order parameters.

        Returns:
            Dict[str, Any]: A dictionary representing the order.
        """
        order = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "order_type": order_type,
            "price": price,
            "timestamp": pd.Timestamp.now(),
        }
        order.update(kwargs)
        return order

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return f"Strategy(name={self.name}, symbols={list(self.datas.keys())}, timeframes={list(next(iter(self.datas.values())).keys())})"
