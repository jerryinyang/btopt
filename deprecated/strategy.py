import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..data.timeframe import Timeframe
from ..indicator.indicator import Indicator
from ..order import Order
from ..parameters import Parameters
from ..sizer import NaiveSizer, Sizer
from ..trade import Trade
from ..types import EngineType
from ..util.ext_decimal import ExtendedDecimal
from ..util.log_config import logger_main
from ..util.metaclasses import PreInitABCMeta
from .helper import Data, DataTimeframe


class StrategyError(Exception):
    pass


class Strategy(metaclass=PreInitABCMeta):
    """
    Abstract base class for trading strategies.

    This class provides a framework for implementing trading strategies, managing
    data, handling orders and trades, and interacting with the trading engine.

    Attributes:
        _id (str): Unique identifier for the strategy.
        name (str): Name of the strategy.
        _engine (EngineType): Reference to the trading engine.
        _parameters (Parameters): Strategy parameters.
        _primary_symbol (Optional[str]): Primary trading symbol.
        _primary_timeframe (Optional[Timeframe]): Primary trading timeframe.
        _strategy_timeframes (List[Timeframe]): List of timeframes used by the strategy.
        _max_bars_back (int): Maximum number of historical bars to store.
        _warmup_period (int): Number of bars required for strategy warm-up.
        _is_warmup_complete (bool): Flag indicating if warm-up is complete.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        primary_timeframe: Optional[Timeframe] = None,
        parameters: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the Strategy.

        Args:
            name (Optional[str]): Name of the strategy. Defaults to None.
            primary_timeframe (Optional[Timeframe]): Primary timeframe for the strategy. Defaults to None.
            parameters (Optional[Dict[str, Any]]): Initial strategy parameters. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._id: str = self._generate_unique_id()
        self._engine: EngineType = None
        self._parameters: Parameters = Parameters(parameters or {})
        self._primary_symbol: Optional[str] = None
        self._primary_timeframe: Optional[Timeframe] = primary_timeframe
        self._strategy_timeframes: List[Timeframe] = []
        self._max_bars_back: int = 500
        self._warmup_period: int = 1
        self._is_warmup_complete: bool = False
        self._risk_percentage: ExtendedDecimal = ExtendedDecimal("0.05")
        self._initialized: bool = False

        self._indicator_configs: List[Dict[str, Any]] = []
        self._indicators: Dict[str, Indicator] = {}

        # Reintegrated data management attributes
        self.name: str = name or self._id
        self.datas: Dict[str, Data] = {}

        # Initialize the position sizer with the default NaivePositionSizer
        self._sizer: Sizer = NaiveSizer()

    # region Initialization and Configuration

    def initialize(self, symbols: List[str], timeframes: List[Timeframe]) -> None:
        """
        Initialize the strategy with symbols and timeframes.

        This method sets up the strategy with the provided symbols and timeframes,
        creating Data objects for each symbol and setting the primary symbol and timeframe.

        Args:
            symbols (List[str]): List of trading symbols.
            timeframes (List[Timeframe]): List of timeframes to use.

        Raises:
            ValueError: If the strategy is already initialized or if no symbols are provided.
        """
        if self._primary_symbol is not None:
            raise ValueError("Strategy is already initialized.")
        if not symbols:
            raise ValueError("At least one symbol must be provided.")

        default_timeframe = min(timeframes)
        self._primary_timeframe = self._primary_timeframe or default_timeframe
        self._strategy_timeframes = timeframes
        self._primary_symbol = symbols[0]

        if self._primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self._primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self._primary_timeframe = default_timeframe

        # Create Data objects for each symbol
        for symbol in symbols:
            self.datas[symbol] = Data(symbol)

        # Create indicator instances
        self._initialize_indicators()

        self._initialized = True
        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self._primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

    def set_engine(self, engine: EngineType) -> None:
        """
        Set the trading engine for the strategy.

        Args:
            engine (EngineType): The trading engine instance.
        """
        self._engine = engine

    def set_sizer(self, sizer: Sizer) -> None:
        """
        Set a new position sizer for the strategy.

        This method allows changing the position sizing method at runtime, providing
        flexibility to adapt to different market conditions or trading requirements.

        Args:
            sizer (Sizer): The new position sizer to be used by the strategy.

        Raises:
            TypeError: If the provided sizer is not an instance of Sizer.
        """
        if not isinstance(sizer, Sizer):
            logger_main.error(f"Invalid position sizer type: {type(sizer)}")
            raise TypeError("sizer must be an instance of Sizer")

        self._sizer = sizer
        logger_main.info(f"Position sizer updated to: {sizer.get_info()['name']}")

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
        Set new strategy parameters and notify the engine.

        Args:
            new_parameters (Dict[str, Any]): A dictionary of new parameter values.
        """
        self._parameters = Parameters(new_parameters)
        if self._engine:
            self._engine.update_strategy_parameters(self._id, new_parameters)
        logger_main.info(
            f"Updated parameters for strategy {self.name}: {new_parameters}"
        )

    @property
    def max_bars_back(self) -> int:
        """
        Get the maximum number of historical bars to store.

        Returns:
            int: The maximum number of bars.
        """
        return self._max_bars_back

    @max_bars_back.setter
    def max_bars_back(self, value: int) -> None:
        """
        Set the maximum number of historical bars to store.

        Args:
            value (int): The new maximum number of bars.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("max_bars_back must be a positive integer.")
        self._max_bars_back = value
        if self._engine:
            self._engine.update_strategy_max_bars_back(self._id, value)
        logger_main.info(f"Updated max_bars_back to {value}")

    @property
    def risk_percentage(self) -> ExtendedDecimal:
        """
        Get the risk percentage for the strategy.

        Returns:
            ExtendedDecimal: The current risk percentage.
        """
        return self._risk_percentage

    @risk_percentage.setter
    def risk_percentage(self, value: float) -> None:
        """
        Set the risk percentage for the strategy.

        Args:
            value (float): The new risk percentage value.

        Raises:
            ValueError: If the value is not between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError("Risk percentage must be between 0 and 1")
        self._risk_percentage = ExtendedDecimal(str(value))
        logger_main.info(f"Updated risk percentage to {value}")

    @property
    def warmup_period(self) -> int:
        """
        Get the warm-up period for the strategy.

        Returns:
            int: The current warm-up period.
        """
        return self._warmup_period

    @warmup_period.setter
    def warmup_period(self, value: int) -> None:
        """
        Set the warm-up period for the strategy.

        Args:
            value (int): The new warm-up period.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("warmup_period must be a positive integer.")
        self._warmup_period = value
        logger_main.info(f"Updated warmup_period to {value}")

    # endregion

    # region Data Management

    @property
    def data(self) -> Data:
        """
        Get the primary data stream for the strategy.

        This property provides access to the Data object for the primary symbol.
        It ensures that the strategy has been properly initialized before
        attempting to access the data.

        Returns:
            Data: The Data object for the primary symbol.

        Raises:
            ValueError: If the strategy hasn't been initialized or if there's no primary symbol set.
        """
        if self._primary_symbol is None:
            raise ValueError("No primary symbol set for the strategy.")
        return self.datas[self._primary_symbol]

    def get_data(self, symbol: Optional[str] = None) -> Data:
        """
        Get the Data object for a specific symbol.

        This method allows access to Data objects for symbols other than the primary symbol.

        Args:
            symbol (Optional[str]): The symbol for which to retrieve the Data object.
                                    If None, returns the primary symbol's Data object.

        Returns:
            Data: The Data object for the specified symbol.

        Raises:
            ValueError: If the strategy hasn't been initialized or if the symbol doesn't exist.
        """
        symbol = symbol or self._primary_symbol
        if symbol not in self.datas:
            raise ValueError(f"No data available for symbol: {symbol}")
        return self.datas[symbol]

    def get_data_timeframe(self, symbol: str, timeframe: Timeframe) -> DataTimeframe:
        """
        Get a DataTimeframe object for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.

        Returns:
            DataTimeframe: The DataTimeframe object for the specified symbol and timeframe.

        Raises:
            ValueError: If the requested symbol is not found.
        """
        if symbol not in self.datas:
            raise ValueError(f"Symbol {symbol} not found in strategy data.")
        return self.datas[symbol][timeframe]

    def get_data_length(
        self, symbol: str, timeframe: Optional[Timeframe] = None
    ) -> int:
        """
        Get the length of available data for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to check.
            timeframe (Optional[Timeframe]): The timeframe to check. If None, uses the primary timeframe.

        Returns:
            int: The number of available data points.

        Raises:
            StrategyError: If the strategy hasn't been initialized or if the symbol doesn't exist.
        """
        if not self._initialized:
            logger_main.log_and_raise(
                StrategyError("Strategy has not been initialized.")
            )

        if symbol not in self.datas:
            logger_main.log_and_raise(
                StrategyError(f"No data available for symbol: {symbol}")
            )

        if timeframe is None:
            timeframe = self._primary_timeframe

        return len(self.datas[symbol][timeframe].close)

    def register_indicator(self, indicator_instance: Indicator, **kwargs: Any) -> None:
        # Check if indicator_instance is a valid subclass of Indicator
        if not issubclass(indicator_instance, Indicator):
            logger_main.log_and_raise(
                TypeError(
                    f"{indicator_instance.__name__} is not a valid Indicator subclass"
                )
            )

        # Check for duplicate indicator name
        name = indicator_instance.name
        if any(config["name"] == name for config in self._indicator_configs):
            logger_main.log_and_raise(
                ValueError(
                    f"An indicator named '{name}' has already been added to this strategy"
                )
            )

        # Process and validate symbols
        symbols = kwargs.get("symbols", [self._primary_symbol])
        if isinstance(symbols, str):
            symbols = [symbols]
        elif not isinstance(symbols, list):
            logger_main.log_and_raise(
                TypeError(
                    f"Expected str or list[str] for `symbols`; received {symbols} of type `{type(symbols)}`"
                )
            )

        # Filter out unrecognized symbols
        valid_symbols = [sym for sym in symbols if sym in self.datas.keys()]
        if len(valid_symbols) != len(symbols):
            unrecognized = set(symbols) - set(valid_symbols)
            logger_main.warning(
                f"Symbols {unrecognized} are not available within the strategy symbols. These symbols will be ignored."
            )

        if not valid_symbols:
            logger_main.warning(
                f"No valid symbols provided for Indicator {name}. Defaulting to primary symbol."
            )
            valid_symbols = [self._primary_symbol]

        # Process and validate timeframes
        timeframes = kwargs.get("timeframes", [self._primary_timeframe])
        if isinstance(timeframes, (str, Timeframe)):
            timeframes = [
                Timeframe(timeframes) if isinstance(timeframes, str) else timeframes
            ]
        elif not isinstance(timeframes, list):
            logger_main.log_and_raise(
                TypeError(
                    f"Expected str, Timeframe, or list[Timeframe] for `timeframes`; received {timeframes} of type `{type(timeframes)}`"
                )
            )

        # Ensure all timeframes are Timeframe objects
        timeframes = [Timeframe(tf) if isinstance(tf, str) else tf for tf in timeframes]

        # Filter out symbols that don't support all specified timeframes
        symbols_to_remove = []
        for sym in valid_symbols:
            for tf in timeframes:
                if tf not in self.datas[sym].timeframes:
                    logger_main.warning(
                        f"Timeframe {tf} is not available for symbol {sym}. This symbol will be removed for this indicator."
                    )
                    symbols_to_remove.append(sym)
                    break

        valid_symbols = [sym for sym in valid_symbols if sym not in symbols_to_remove]

        if not valid_symbols:
            logger_main.warning(
                f"No symbols support all specified timeframes for Indicator {name}. Defaulting to primary symbol and timeframe."
            )
            valid_symbols = [self._primary_symbol]
            timeframes = [self._primary_timeframe]

        kwargs["symbols"] = valid_symbols
        kwargs["timeframes"] = timeframes

        # Store the validated configuration
        self._indicator_configs.append(
            {"instance": indicator_instance, "kwargs": kwargs}
        )
        logger_main.info(
            f"Registered  indicator '{name}' in strategy '{self.name}' with symbols {valid_symbols} and timeframes {timeframes}"
        )

    def _initialize_indicators(self) -> None:
        """
        Initialize indicator instances based on stored configurations.
        """

        for config in self._indicator_configs:
            indicator_instance: Indicator = config["instance"]
            kwargs: dict = config["kwargs"]
            kwargs["strategy"] = self

            indicator_instance._initialize_indicator(**kwargs)

            self._indicators[indicator_instance.name] = indicator_instance
            logger_main.info(
                f"Created indicator instance '{indicator_instance.name}' in strategy '{self.name}'"
            )

    def get_indicator_output(self, indicator_name: str, output_name: str) -> Any:
        """
        Get the output of a specific indicator.

        Args:
            indicator_name (str): The name of the indicator.
            output_name (str): The name of the output to retrieve.

        Returns:
            Any: The value of the specified indicator output.

        Raises:
            KeyError: If the specified indicator does not exist.
            AttributeError: If the specified output does not exist for the indicator.
        """
        if indicator_name not in self._indicators:
            raise KeyError(
                f"Indicator '{indicator_name}' does not exist in this strategy"
            )

        return self._indicators[indicator_name].get(output_name)

    # endregion

    # region Order Management

    def buy(
        self,
        symbol: str,
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Order], List[Order]]:
        """
        Create a buy order request.

        Args:
            symbol (str): The symbol to buy.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            stop_loss (Optional[float]): The stop-loss price for the order.
            take_profit (Optional[float]): The take-profit price for the order.
            **kwargs: Additional order parameters.

        Returns:
            Tuple[Optional[Order], List[Order]]: The created parent Order object and a list of child orders,
            or (None, []) if the order creation failed.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.create_order(
            self._id,
            symbol,
            Order.Direction.LONG,
            size,
            Order.ExecType.MARKET if price is None else Order.ExecType.LIMIT,
            price,
            stop_loss,
            take_profit,
            **kwargs,
        )

    def sell(
        self,
        symbol: str,
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Order], List[Order]]:
        """
        Create a sell order request.

        Args:
            symbol (str): The symbol to sell.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            stop_loss (Optional[float]): The stop-loss price for the order.
            take_profit (Optional[float]): The take-profit price for the order.
            **kwargs: Additional order parameters.

        Returns:
            Tuple[Optional[Order], List[Order]]: The created parent Order object and a list of child orders,
            or (None, []) if the order creation failed.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.create_order(
            self._id,
            symbol,
            Order.Direction.SHORT,
            size,
            Order.ExecType.MARKET if price is None else Order.ExecType.LIMIT,
            price,
            stop_loss,
            take_profit,
            **kwargs,
        )

    def close(self, symbol: Optional[str] = None, size: Optional[float] = None) -> bool:
        """
        Close positions for this strategy, optionally for a specific symbol and size.

        Args:
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.
            size (Optional[float]): The size of the position to close. If None, close the entire position.

        Returns:
            bool: True if the closing operation was successful, False otherwise.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.close_positions(self._id, symbol, size)

    def cancel(self, order: Order) -> bool:
        """
        Request cancellation of an existing order.

        Args:
            order (Order): The order to cancel.

        Returns:
            bool: True if the cancellation request was successful, False otherwise.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.cancel_order(self._id, order)

    # endregion

    # region Position and Risk Management

    def get_current_position(
        self,
        symbol: str,
    ) -> float:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            float: The current position size (positive for long, negative for short, 0 for no position).

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.get_position_size(symbol)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal] = None,
    ) -> ExtendedDecimal:
        """
        Calculate the position size using the current position sizer.

        This method delegates the position size calculation to the current position sizer,
        providing a consistent interface regardless of the specific sizing method in use.

        Args:
            symbol (str): The trading symbol for which to calculate the position size.
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade (e.g., stop loss).

        Returns:
            ExtendedDecimal: The calculated position size.

        Raises:
            ValueError: If the input parameters are invalid.
            RuntimeError: If the trading engine is not set or the sizer is not set.
        """
        if self._engine is None:
            logger_main.error("Trading engine is not set")
            raise RuntimeError(
                "Trading engine must be set before calculating position size"
            )

        if not hasattr(self, "_sizer") or self._sizer is None:
            logger_main.error("Position sizer is not set")
            raise RuntimeError(
                "Position sizer must be set before calculating position size"
            )

        try:
            position_size = self._sizer.calculate_position_size(
                strategy=self,
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
            )
            logger_main.info(f"Calculated position size: {position_size}")
            return position_size
        except ValueError as e:
            logger_main.error(f"Position size calculation failed: {str(e)}")
            raise

    def validate_position_size(
        self,
        symbol: str,
        position_size: ExtendedDecimal,
        min_position_size: Optional[ExtendedDecimal] = None,
        max_position_size: Optional[ExtendedDecimal] = None,
    ) -> ExtendedDecimal:
        """
        Validate and adjust the calculated position size based on defined limits.

        This method ensures that the calculated position size falls within acceptable limits,
        adjusting it if necessary.

        Args:
            symbol (str): The symbol the position size is calculated for.
            position_size (ExtendedDecimal): The calculated position size to validate.
            min_position_size (Optional[ExtendedDecimal]): The minimum allowed position size. Defaults to None.
            max_position_size (Optional[ExtendedDecimal]): The maximum allowed position size. Defaults to None.

        Returns:
            ExtendedDecimal: The validated and potentially adjusted position size.

        Raises:
            ValueError: If the position size is 0 or negative, or if it's below the minimum allowed size.
        """
        if position_size <= 0:
            raise ValueError("Calculated position size must be greater than 0.")

        if min_position_size is not None and position_size < min_position_size:
            logger_main.warning(
                f"Calculated position size for {symbol} is below the minimum allowed size. Adjusting to minimum."
            )
            return min_position_size

        if max_position_size is not None:
            return min(position_size, max_position_size)

        return position_size

    # endregion

    # region Performance Tracking

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the strategy.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.get_strategy_performance(self._id)

    # endregion

    # region Abstract Methods

    @abstractmethod
    def _on_data(self) -> None:
        """
        Internal method called by the engine to process new data.

        This method updates indicators and checks if the warm-up period is complete
        before calling the user-defined on_data method.
        """
        # Update indicators
        for indicator in self._indicators.values():
            indicator._on_data()

        if self._check_warmup_period():
            self.on_data()

    def on_order_update(self, order: Order) -> None:
        """
        Handle updates to orders created by this strategy.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to order updates.

        Args:
            order (Order): The updated Order object.
        """
        pass

    def on_trade_update(self, trade: Trade) -> None:
        """
        Handle updates to trades associated with this strategy.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to trade updates.

        Args:
            trade (Trade): The updated Trade object.
        """
        pass

    # endregion

    # region Utility Methods

    def log(self, message: str) -> None:
        """
        Log a message through the engine's logging system.

        Args:
            message (str): The message to log.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        self._engine.log_strategy_activity(self._id, message)

    def _generate_unique_id(self) -> str:
        """
        Generate a unique identifier for the strategy.

        Returns:
            str: A unique identifier.
        """
        return str(uuid.uuid4())

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return (
            f"Strategy(name={self.name}, id={self._id}, "
            f"primary_symbol={self._primary_symbol}, "
            f"primary_timeframe={self._primary_timeframe})"
        )

    def __len__(self) -> int:
        """
        Get the length of stored data for the primary symbol and timeframe.

        Returns:
            int: The number of available data points for the primary symbol and timeframe.

        Raises:
            ValueError: If the strategy hasn't been initialized or if there's no primary symbol set.
        """
        if not self._primary_symbol or not self._primary_timeframe:
            raise ValueError("Strategy has not been properly initialized.")
        return self.get_data_length(self._primary_symbol, self._primary_timeframe)

    # endregion

    # region Backward Compatibility Methods

    def get_trades_for_strategy(self) -> List[Trade]:
        """
        Get all trades for this strategy.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.get_trades_for_strategy(self._id)

    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders for this strategy.

        Returns:
            List[Order]: A list of pending Order objects.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.get_pending_orders_for_strategy(self._id)

    def get_positions(self) -> Dict[str, float]:
        """
        Get all current positions for this strategy.

        Returns:
            Dict[str, float]: A dictionary of current positions, keyed by symbol.

        Raises:
            ValueError: If the strategy is not connected to an engine.
        """
        if not self._engine:
            raise ValueError("Strategy is not connected to an engine.")
        return self._engine.get_positions_for_strategy(self._id)

    # endregion

    # region Internal Methodss

    def _on_data(self) -> None:
        """
        Internal method called by the engine to process new data.

        This method updates indicators and checks if the warm-up period is complete
        before calling the user-defined on_data method.
        """
        # Update indicators
        for indicator in self._indicators.values():
            indicator._on_data()

        if self._check_warmup_period():
            self.on_data()

    def _check_warmup_period(self) -> bool:
        """
        Check if the warm-up period is complete for all symbols and timeframes.

        Returns:
            bool: True if the warm-up period is complete, False otherwise.
        """
        if self._is_warmup_complete:
            return True

        for symbol, data in self.datas.items():
            for timeframe in self._strategy_timeframes:
                if timeframe not in data.timeframes:
                    logger_main.warning(
                        f"Timeframe `{repr(timeframe)}` was not found for symbol `{symbol}`. "
                        f"Available timeframes are {self._strategy_timeframes}"
                    )
                    return False
                elif len(data[timeframe]) < self._warmup_period:
                    return False

        self._is_warmup_complete = True
        logger_main.info(f"Strategy {self._id} warm-up is complete.")
        return True

    # endregion
