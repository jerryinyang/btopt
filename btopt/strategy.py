import uuid
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from .data.manager_price import PriceDataManager
from .data.timeframe import Timeframe
from .indicator import Indicator
from .order import (
    BracketGroup,
    BracketOrderDetails,
    OCOGroup,
    OCOOrderDetails,
    Order,
    OrderDetails,
)
from .parameters import Parameters
from .sizer import NaiveSizer, Sizer
from .trade import Trade
from .types import EngineType, PortfolioType
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main
from .util.metaclasses import PreInitABCMeta


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
        self._portfolio: PortfolioType = None
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
        self.datas: Dict[str, PriceDataManager] = {}

        # Initialize the position sizer with the default NaivePositionSizer
        self._sizer: Sizer = NaiveSizer()

    # region Initialization and Configuration

    def initialize(self, symbols: List[str], timeframes: List[Timeframe]) -> None:
        """
        Initialize the strategy with symbols and timeframes.

        This method sets up the strategy with the provided symbols and timeframes,
        creating PriceDataManager objects for each symbol and setting the primary symbol and timeframe.

        Args:
            symbols (List[str]): List of trading symbols.
            timeframes (List[Timeframe]): List of timeframes to use.

        Raises:
            ValueError: If the strategy is already initialized or if no symbols are provided.
        """

        if self._primary_symbol is not None:
            logger_main.log_and_raise(ValueError("Strategy is already initialized."))
        if not symbols:
            logger_main.log_and_raise(
                ValueError("At least one symbol must be provided.")
            )

        default_timeframe = min(timeframes)
        self._primary_timeframe = self._primary_timeframe or default_timeframe
        self._strategy_timeframes = timeframes
        self._primary_symbol = symbols[0]

        if self._primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self._primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self._primary_timeframe = default_timeframe

        if self._engine.portfolio is None:
            logger_main.log_and_raise(
                ValueError("Engine's portfolio is not initialized.")
            )
        self._portfolio = self._engine.portfolio

        # Create PriceDataManager objects for each symbol
        for symbol in symbols:
            self.datas[symbol] = PriceDataManager(symbol)

        self._initialized = True

        # Create indicator instances
        self._initialize_indicators()

        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self._primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

    def set_engine(self, engine: EngineType) -> None:
        """
        Set the trading engine and portfolio for the strategy.

        Args:
            engine (EngineType): The trading engine instance.

        Raises:
            ValueError: If the engine's portfolio is not initialized.
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
            logger_main.log_and_raise(
                TypeError(
                    "Invalid position sizer type: {type(sizer)}. sizer must be an instance of Sizer"
                )
            )

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
            logger_main.log_and_raise(
                ValueError("max_bars_back must be a positive integer.")
            )
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
            logger_main.log_and_raise(
                ValueError("Risk percentage must be between 0 and 1")
            )
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
            logger_main.log_and_raise(
                ValueError("warmup_period must be a positive integer.")
            )
        self._warmup_period = value
        logger_main.info(f"Updated warmup_period to {value}")

    # endregion

    # region Data Management

    @property
    def data(self) -> PriceDataManager:
        """
        Get the primary data stream for the strategy.

        This property provides access to the PriceDataManager object for the primary symbol.
        It ensures that the strategy has been properly initialized before
        attempting to access the data.

        Returns:
            PriceDataManager: The PriceDataManager object for the primary symbol.

        Raises:
            ValueError: If the strategy hasn't been initialized or if there's no primary symbol set.
        """
        if self._primary_symbol is None:
            logger_main.log_and_raise(
                ValueError("No primary symbol set for the strategy.")
            )
        return self.datas[self._primary_symbol]

    def get_data(self, symbol: Optional[str] = None) -> PriceDataManager:
        """
        Get the PriceDataManager object for a specific symbol.

        This method allows access to PriceDataManager objects for symbols other than the primary symbol.

        Args:
            symbol (Optional[str]): The symbol for which to retrieve the PriceDataManager object.
                                    If None, returns the primary symbol's PriceDataManager object.

        Returns:
            PriceDataManager: The PriceDataManager object for the specified symbol.

        Raises:
            ValueError: If the strategy hasn't been initialized or if the symbol doesn't exist.
        """
        symbol = symbol or self._primary_symbol
        if symbol not in self.datas:
            logger_main.log_and_raise(
                ValueError(f"No data available for symbol: {symbol}")
            )
        return self.datas[symbol]

    def get_data_timeframe(self, symbol: str, timeframe: Timeframe) -> PriceDataManager:
        """
        Get a PriceDataManager object for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.

        Returns:
            PriceDataManager: The PriceDataManager object for the specified symbol and timeframe.

        Raises:
            ValueError: If the requested symbol is not found.
        """
        if symbol not in self.datas:
            logger_main.log_and_raise(
                ValueError(f"Symbol {symbol} not found in strategy data.")
            )
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

    def add_indicator(self, indicator_instance: Indicator, **kwargs: Any):
        # Check if indicator_instance is a valid subclass of Indicator
        if not isinstance(indicator_instance, Indicator):
            logger_main.log_and_raise(
                TypeError(
                    f"{indicator_instance.__name__} is not a valid Indicator subclass"
                )
            )

        # Check for duplicate indicator name
        name = indicator_instance.name
        logger_main.warning(self._indicator_configs)
        if any(config["instance"].name == name for config in self._indicator_configs):
            logger_main.log_and_raise(
                ValueError(
                    f"An indicator named '{name}' has already been added to this strategy"
                )
            )

        # Store the validated configuration
        self._indicator_configs.append(
            {"instance": indicator_instance, "kwargs": kwargs}
        )
        logger_main.info(f"Added indicator '{name}' to strategy '{self.name}'.")

    def _register_indicator(
        self, indicator_instance: Indicator, kwargs: Any
    ) -> Tuple[Indicator, Dict]:
        name = indicator_instance.name

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
        valid_symbols = [symbol for symbol in symbols if symbol in self.datas.keys()]
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

        # Process and validate timeframe
        timeframe = kwargs.get("timeframe", self._primary_timeframe)
        if isinstance(timeframe, (str, Timeframe)):
            timeframe = (
                Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
            )
        else:
            logger_main.log_and_raise(
                TypeError(
                    f"Expected str or Timeframe for `timeframe`; received {timeframe} of type `{type(timeframe)}`"
                )
            )

        if timeframe not in self._strategy_timeframes:
            logger_main.warning(
                f"Timeframe provided for Indicator {name} not found within the associated strategy. Defaulting to primary timeframe."
            )
            timeframe = self._primary_timeframe

        kwargs["symbols"] = valid_symbols
        kwargs["timeframe"] = timeframe
        kwargs["strategy"] = self

        return indicator_instance, kwargs

    def _initialize_indicators(self) -> None:
        """
        Initialize indicator instances based on stored configurations.
        """

        for config in self._indicator_configs:
            indicator_instance: Indicator = config["instance"]
            kwargs: dict = config["kwargs"]

            indicator_instance, kwargs = self._register_indicator(
                indicator_instance, kwargs
            )
            indicator_instance._initialize_indicator(**kwargs)

            self._indicators[indicator_instance.name] = indicator_instance
            logger_main.info(
                f"Created and registered indicator instance '{indicator_instance.name}' to strategy '{self.name}'"
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
            logger_main.log_and_raise(
                KeyError(
                    f"Indicator '{indicator_name}' does not exist in this strategy"
                )
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
    ) -> Union[Order, Tuple[List[Order], BracketGroup]]:
        """
        Create a buy order request, automatically creating a bracket order if stop_loss or take_profit is specified.

        Args:
            symbol (str): The symbol to buy.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            stop_loss (Optional[float]): The stop-loss price for the order.
            take_profit (Optional[float]): The take-profit price for the order.
            **kwargs: Additional order parameters.

        Returns:
            Union[Order, Tuple[List[Order], BracketGroup]]:
                - A single Order object if no stop_loss or take_profit is specified.
                - A tuple containing the list of orders and the BracketGroup if stop_loss or take_profit is specified.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        order_details = OrderDetails(
            ticker=symbol,
            direction=Order.Direction.LONG,
            size=ExtendedDecimal(str(size)),
            price=ExtendedDecimal(str(price)) if price is not None else None,
            exectype=Order.ExecType.LIMIT
            if price is not None
            else Order.ExecType.MARKET,
            timestamp=self._engine._current_timestamp,
            timeframe=self._primary_timeframe,
            strategy_id=self._id,
            exit_loss=ExtendedDecimal(str(stop_loss))
            if stop_loss is not None
            else None,
            exit_profit=ExtendedDecimal(str(take_profit))
            if take_profit is not None
            else None,
            **kwargs,
        )

        if stop_loss is None and take_profit is None:
            return self._portfolio.create_order(order_details)
        else:
            return self._portfolio.create_bracket_order(
                BracketOrderDetails(
                    entry_order=order_details,
                    take_profit_order=OrderDetails(
                        ticker=symbol,
                        direction=Order.Direction.SHORT,
                        size=ExtendedDecimal(str(size)),
                        price=ExtendedDecimal(str(take_profit)),
                        exectype=Order.ExecType.LIMIT,
                        timestamp=self._engine._current_timestamp,
                        timeframe=self._primary_timeframe,
                        strategy_id=self._id,
                    ),
                    stop_loss_order=OrderDetails(
                        ticker=symbol,
                        direction=Order.Direction.SHORT,
                        size=ExtendedDecimal(str(size)),
                        price=ExtendedDecimal(str(stop_loss)),
                        exectype=Order.ExecType.STOP,
                        timestamp=self._engine._current_timestamp,
                        timeframe=self._primary_timeframe,
                        strategy_id=self._id,
                    ),
                )
            )

    def sell(
        self,
        symbol: str,
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[Order, Tuple[List[Order], BracketGroup]]:
        """
        Create a sell order request, automatically creating a bracket order if stop_loss or take_profit is specified.

        Args:
            symbol (str): The symbol to sell.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            stop_loss (Optional[float]): The stop-loss price for the order.
            take_profit (Optional[float]): The take-profit price for the order.
            **kwargs: Additional order parameters.

        Returns:
            Union[Order, Tuple[List[Order], BracketGroup]]:
                - A single Order object if no stop_loss or take_profit is specified.
                - A tuple containing the list of orders and the BracketGroup if stop_loss or take_profit is specified.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        order_details = OrderDetails(
            ticker=symbol,
            direction=Order.Direction.SHORT,
            size=ExtendedDecimal(str(size)),
            price=ExtendedDecimal(str(price)) if price is not None else None,
            exectype=Order.ExecType.LIMIT
            if price is not None
            else Order.ExecType.MARKET,
            timestamp=self._engine._current_timestamp,
            timeframe=self._primary_timeframe,
            strategy_id=self._id,
            exit_loss=ExtendedDecimal(str(stop_loss))
            if stop_loss is not None
            else None,
            exit_profit=ExtendedDecimal(str(take_profit))
            if take_profit is not None
            else None,
            **kwargs,
        )

        if stop_loss is None and take_profit is None:
            return self._portfolio.create_order(order_details)
        else:
            return self._portfolio.create_bracket_order(
                BracketOrderDetails(
                    entry_order=order_details,
                    take_profit_order=OrderDetails(
                        ticker=symbol,
                        direction=Order.Direction.LONG,
                        size=ExtendedDecimal(str(size)),
                        price=ExtendedDecimal(str(take_profit)),
                        exectype=Order.ExecType.LIMIT,
                        timestamp=self._engine._current_timestamp,
                        timeframe=self._primary_timeframe,
                        strategy_id=self._id,
                    ),
                    stop_loss_order=OrderDetails(
                        ticker=symbol,
                        direction=Order.Direction.LONG,
                        size=ExtendedDecimal(str(size)),
                        price=ExtendedDecimal(str(stop_loss)),
                        exectype=Order.ExecType.STOP,
                        timestamp=self._engine._current_timestamp,
                        timeframe=self._primary_timeframe,
                        strategy_id=self._id,
                    ),
                )
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
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.close_positions(
            strategy_id=self._id, symbol=symbol, size=size
        )

    def cancel(self, order_id: str) -> bool:
        """
        Request cancellation of an existing order.

        Args:
            order_id (str): The ID of the order to cancel.

        Returns:
            bool: True if the cancellation request was successful, False otherwise.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.cancel_order(order_id)

    def create_oco_order(
        self,
        symbol: str,
        direction: Order.Direction,
        size: float,
        limit_price: float,
        stop_price: float,
        **kwargs: Any,
    ) -> Tuple[List[Order], OCOGroup]:
        """
        Create an OCO (One-Cancels-the-Other) order.

        Args:
            symbol (str): The symbol to trade.
            direction (Order.Direction): The direction of the trade (LONG or SHORT).
            size (float): The size of the order.
            limit_price (float): The price for the limit order.
            stop_price (float): The price for the stop order.
            **kwargs: Additional order parameters.

        Returns:
            Tuple[List[Order], OCOGroup]: A tuple containing the list of created orders and the OCO group.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        oco_details = OCOOrderDetails(
            limit_order=OrderDetails(
                ticker=symbol,
                direction=direction,
                size=ExtendedDecimal(str(size)),
                price=ExtendedDecimal(str(limit_price)),
                exectype=Order.ExecType.LIMIT,
                timestamp=self._engine._current_timestamp,
                timeframe=self._primary_timeframe,
                strategy_id=self._id,
                **kwargs,
            ),
            stop_order=OrderDetails(
                ticker=symbol,
                direction=direction,
                size=ExtendedDecimal(str(size)),
                price=ExtendedDecimal(str(stop_price)),
                exectype=Order.ExecType.STOP,
                timestamp=self._engine._current_timestamp,
                timeframe=self._primary_timeframe,
                strategy_id=self._id,
                **kwargs,
            ),
        )

        return self._portfolio.create_oco_order(oco_details)

    # endregion

    # region Position and Risk Management

    def get_trades_for_strategy(self) -> List[Trade]:
        """
        Get all trades for this strategy.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.get_trades_for_strategy(self._id)

    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders for this strategy.

        Returns:
            List[Order]: A list of pending Order objects.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.get_pending_orders_for_strategy(self._id)

    def get_positions(self) -> Dict[str, float]:
        """
        Get all current positions for this strategy.

        Returns:
            Dict[str, float]: A dictionary of current positions, keyed by symbol.

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.get_positions_for_strategy(self._id)

    def get_current_position(self, symbol: str) -> float:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            float: The current position size (positive for long, negative for short, 0 for no position).

        Raises:
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.get_position_size(symbol)

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: ExtendedDecimal,
        exit_price: Optional[ExtendedDecimal] = None,
    ) -> ExtendedDecimal:
        """
        Calculate the position size using the current position sizer and portfolio's risk manager.

        This method delegates the position size calculation to the current position sizer,
        providing a consistent interface regardless of the specific sizing method in use.

        Args:
            symbol (str): The trading symbol for which to calculate the position size.
            entry_price (ExtendedDecimal): The entry price for the trade.
            exit_price (Optional[ExtendedDecimal]): The exit price for the trade (e.g., stop loss).

        Returns:
            ExtendedDecimal: The calculated position size.

        Raises:
            ValueError: If the input parameters are invalid or if the strategy is not connected to a portfolio.
            RuntimeError: If the sizer is not set.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        if not hasattr(self, "_sizer") or self._sizer is None:
            logger_main.log_and_raise(
                RuntimeError(
                    "Position sizer must be set before calculating position size"
                )
            )

        try:
            risk_amount = self._portfolio.calculate_risk_amount(symbol)
            position_size = self._sizer.calculate_position_size(
                entry_price=entry_price,
                exit_price=exit_price,
                risk_amount=risk_amount * self._risk_percentage,
            )
            logger_main.info(f"Calculated position size: {position_size}")
            return position_size
        except ValueError as e:
            logger_main.log_and_raise(f"Position size calculation failed: {str(e)}")
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
            logger_main.log_and_raise(
                ValueError("Calculated position size must be greater than 0.")
            )

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
            ValueError: If the strategy is not connected to a portfolio.
        """
        if not hasattr(self, "_portfolio") or self._portfolio is None:
            logger_main.log_and_raise(
                ValueError("Strategy is not connected to a portfolio.")
            )

        return self._portfolio.get_strategy_performance(self._id)

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

        Note:
            This method now uses the main logger directly instead of going through the engine.
        """
        logger_main.info(f"Strategy {self._id}: {message}")

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
            logger_main.log_and_raise(
                ValueError("Strategy has not been properly initialized.")
            )
        return self.get_data_length(self._primary_symbol, self._primary_timeframe)

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
