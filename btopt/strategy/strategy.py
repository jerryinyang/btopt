import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main
from ..order import Order
from ..parameters import Parameters
from ..trade import Trade
from .helper import BarManager, Data


class StrategyError(Exception):
    """
    Abstract base class for trading strategies.

    This class provides a comprehensive framework for creating and managing trading strategies
    in a financial backtesting or live trading system. It defines the interface and common
    functionality that all strategy implementations should follow, ensuring consistency
    and modularity across different trading approaches.

    The Strategy class is designed to work in conjunction with an Engine class, which
    handles the overall execution of the trading system, and a Portfolio class, which
    manages positions and risk. It provides methods for handling market data, generating
    trading signals, managing orders, and responding to trade events.

    Key Features:
    1. Data Management: Stores and provides access to historical and real-time market data
       for multiple symbols and timeframes.
    2. Signal Generation: Abstract method for implementing strategy-specific logic to
       generate trading signals based on market data.
    3. Order Management: Methods for creating, modifying, and canceling orders.
    4. Event Handling: Abstract methods for responding to order and trade events.
    5. Position Sizing: Utility method for calculating position sizes based on risk parameters.
    6. Parameter Management: Flexible system for managing and updating strategy parameters.

    Attributes:
        name (str): The name of the strategy. Used for identification and logging.
        primary_timeframe (Optional[Timeframe]): The primary timeframe used by the strategy
            for signal generation and decision making.
        _datas (Dict[str, Dict[Timeframe, Data]]): A nested dictionary storing market data
            for each symbol and timeframe combination. Provides efficient access to
            historical and real-time data.
        _bar_manager (BarManager): A utility object for storing and managing price bars.
            Provides additional functionality for data access and manipulation.
        _initialized (bool): A flag indicating whether the strategy has been properly
            initialized with symbols and timeframes.
        _primary_symbol (Optional[str]): The main trading symbol for the strategy.
            Used as a default when accessing data or placing orders.
        _engine: A reference to the Engine object that manages this strategy.
            Provides access to broader system functionality like order execution
            and portfolio management.
        _parameters (Parameters): An object managing the strategy's parameters.
            Allows for easy setting, getting, and validation of strategy parameters.
        _id (str): A unique identifier for the strategy instance. Used for
            distinguishing between multiple instances of the same strategy.

    Usage:
        To create a specific trading strategy, subclass this Strategy class and
        implement the abstract methods (generate_signals, on_order, on_trade).
        The subclass can then be instantiated and added to a trading engine for
        backtesting or live trading.

    Example:
        class SimpleMovingAverageCrossover(Strategy):
            def __init__(self, name, parameters):
                super().__init__(name, parameters)
                self.short_period = self.parameters.get('short_period', 10)
                self.long_period = self.parameters.get('long_period', 20)

            def generate_signals(self, bar):
                # Implementation of the moving average crossover logic
                ...

            def on_order(self, order):
                # Handle order updates
                ...

            def on_trade(self, trade):
                # Handle trade updates
                ...

    Note:
        This class is designed to be flexible and extensible. When subclassing,
        ensure that all abstract methods are implemented and consider overriding
        other methods like on_bar for custom behavior.
    """

    pass


def generate_unique_id() -> str:
    """Generate a unique identifier for the strategy."""
    return str(uuid.uuid4())


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    This class provides a framework for creating specific trading strategies.
    It defines the interface and common functionality that all strategy
    implementations should follow.

    Attributes:
        name (str): The name of the strategy.
        primary_timeframe (Optional[Timeframe]): The primary timeframe for the strategy.
        _datas (Dict[str, Dict[Timeframe, Data]]): Nested dictionary to store data for each symbol and timeframe.
        _bar_manager (BarManager): Manager for storing and accessing Bar objects.
        _initialized (bool): Flag indicating whether the strategy has been initialized.
        _primary_symbol (Optional[str]): The primary symbol for the strategy.
        _engine: Reference to the Engine object.
        _parameters (Parameters): Strategy parameters managed by the Parameters class.
        _id (str): Unique identifier for the strategy.
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
        self._parameters: Parameters = Parameters(parameters or {})
        self.primary_timeframe: Optional[Timeframe] = primary_timeframe
        self._datas: Dict[str, Dict[Timeframe, Data]] = defaultdict(dict)
        self._bar_manager: BarManager = BarManager()
        self._initialized: bool = False
        self._primary_symbol: Optional[str] = None
        self._engine = None
        self._strategy_timeframes = {}
        self._id: str = generate_unique_id()

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

        Raises:
            StrategyError: If the strategy is already initialized.
        """
        if self._initialized:
            raise StrategyError("Strategy is already initialized.")

        if self.primary_timeframe is None:
            self.primary_timeframe = default_timeframe
        elif self.primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self.primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self.primary_timeframe = default_timeframe

        for symbol in symbols:
            for timeframe in timeframes:
                self._datas[symbol][timeframe] = Data(symbol, timeframe)

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
            StrategyError: If the strategy hasn't been initialized or if there's no primary symbol.
        """
        if not self._initialized:
            raise StrategyError("Strategy has not been initialized.")
        if self._primary_symbol is None:
            raise StrategyError("No primary symbol set for the strategy.")
        try:
            return self._datas[self._primary_symbol][self.primary_timeframe]
        except KeyError:
            raise StrategyError(
                f"Data for primary symbol {self._primary_symbol} and timeframe {self.primary_timeframe} not found."
            )

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
        self._datas[symbol][timeframe].add_bar(bar)
        self._bar_manager.add_bar(bar)

        # Generate signals and create orders
        signals = self.generate_signals(bar)
        for signal in signals:
            if signal["action"] == "BUY":
                order = self.buy(signal["symbol"], signal["size"], signal.get("price"))
            elif signal["action"] == "SELL":
                order = self.sell(signal["symbol"], signal["size"], signal.get("price"))

            if order:
                self.on_order(order)

        # Check for any trade updates
        if self._engine:
            trades = self._engine.get_trades_for_strategy(self._id)
            for trade in trades:
                self.on_trade(trade)

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
            return self._datas[key][self.primary_timeframe]
        elif isinstance(key, tuple) and len(key) == 2:
            symbol, timeframe = key
            return self._datas[symbol][timeframe]
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
        return self._datas[symbol][timeframe]

    def buy(
        self, symbol: str, size: float, price: Optional[float] = None, **kwargs: Any
    ) -> Optional[Order]:
        """
        Create a buy order.

        Args:
            symbol (str): The symbol to buy.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            **kwargs: Additional order parameters.

        Returns:
            Optional[Order]: The created Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        return self._engine.create_order(
            self, symbol, Order.Direction.LONG, size, price, **kwargs
        )

    def sell(
        self, symbol: str, size: float, price: Optional[float] = None, **kwargs: Any
    ) -> Optional[Order]:
        """
        Create a sell order.

        Args:
            symbol (str): The symbol to sell.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            **kwargs: Additional order parameters.

        Returns:
            Optional[Order]: The created Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        return self._engine.create_order(
            self, symbol, Order.Direction.SHORT, size, price, **kwargs
        )

    def cancel(self, order: Order) -> bool:
        """
        Cancel an existing order.

        Args:
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        return self._engine.cancel_order(self, order)

    def close(self, symbol: Optional[str] = None) -> bool:
        """
        Close all positions for this strategy, or for a specific symbol if provided.

        Args:
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.

        Returns:
            bool: True if the closing operation was successful, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        return self._engine.close_positions(self, symbol)

    @abstractmethod
    def on_order(self, order: Order) -> None:
        """
        Handle order events.

        This method is called when there's an update to an order created by this strategy.

        Args:
            order (Order): The updated Order object.
        """
        pass

    @abstractmethod
    def on_trade(self, trade: Trade) -> None:
        """
        Handle trade events.

        This method is called when there's an update to a trade associated with this strategy.

        Args:
            trade (Trade): The updated Trade object.
        """
        pass

    def calculate_position_size(
        self, symbol: str, risk_percent: float, stop_loss: float
    ) -> float:
        """
        Calculate the position size based on risk percentage and stop loss.

        Args:
            symbol (str): The symbol to trade.
            risk_percent (float): The percentage of account to risk on this trade.
            stop_loss (float): The stop loss price.

        Returns:
            float: The calculated position size.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if the risk per share is zero.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")

        account_value = self._engine.get_account_value()
        risk_amount = account_value * (risk_percent / 100)
        current_price = self._datas[symbol][self.primary_timeframe].close[-1]
        risk_per_share = abs(current_price - stop_loss)

        if risk_per_share == 0:
            raise StrategyError(
                "Risk per share is zero. Cannot calculate position size."
            )

        position_size = risk_amount / risk_per_share
        return position_size

    def get_current_position(self, symbol: str) -> float:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            float: The current position size (positive for long, negative for short, 0 for no position).

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")

        return self._engine.get_position_size(self._id, symbol)

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return f"Strategy(name={self.name}, id={self._id}, symbols={list(self._datas.keys())}, timeframes={list(next(iter(self._datas.values())).keys())})"
