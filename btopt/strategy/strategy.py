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
    pass


def generate_unique_id() -> str:
    """Generate a unique identifier for the strategy."""
    return str(uuid.uuid4())


class Strategy(ABC):
    """
    A comprehensive abstract base class for implementing trading strategies.

    This class provides a robust framework for creating and managing trading strategies
    within a backtesting or live trading environment. It encapsulates core functionality
    for data handling, order management, position tracking, risk management, and
    performance analysis. The class is designed to be flexible and extensible,
    allowing for the implementation of a wide range of trading strategies across
    various financial instruments and timeframes.

    Key Features:
    1. Multi-timeframe and multi-symbol support: Capable of handling data and
       executing trades across multiple symbols and timeframes simultaneously.
    2. Sophisticated order management: Supports various order types including
       market, limit, stop, and OCO (One-Cancels-Other) orders.
    3. Advanced position management: Tracks positions, handles partial fills,
       and supports pyramiding and scaling in/out of positions.
    4. Risk management tools: Includes methods for position sizing, stop-loss
       and take-profit management, and enforcing risk limits.
    5. Performance tracking: Provides methods to calculate and track key
       performance metrics for the strategy.
    6. Event-driven architecture: Utilizes callback methods to handle market
       events, order updates, and trade updates.

    The class is designed to work in conjunction with a backtesting engine,
    which is responsible for simulating market conditions, executing orders,
    and managing the overall flow of the backtest.

    Attributes:
        name (str): The name of the strategy.
        _parameters (Parameters): The strategy's parameters, allowing for easy
            configuration and optimization.
        primary_timeframe (Optional[Timeframe]): The primary timeframe used by
            the strategy for decision making.
        _datas (Dict[str, Dict[Timeframe, Data]]): A nested dictionary storing
            price and indicator data for different symbols and timeframes.
        _bar_manager (BarManager): Manages and aligns bar data across different
            timeframes.
        _initialized (bool): Flag indicating whether the strategy has been
            properly initialized.
        _primary_symbol (Optional[str]): The main trading symbol for the strategy.
        _engine (Any): Reference to the backtesting engine, used for interacting
            with the simulated market environment.
        _strategy_timeframes (Dict[Timeframe, List[str]]): Mapping of timeframes
            to the symbols traded on those timeframes.
        _id (str): Unique identifier for the strategy instance.
        _positions (Dict[str, float]): Current positions held by the strategy,
            keyed by symbol.
        _pending_orders (List[Order]): List of orders that have been submitted
            but not yet filled or cancelled.
        _open_trades (Dict[str, List[Trade]]): Currently open trades, organized
            by symbol.
        _closed_trades (List[Trade]): Historical record of closed trades.

    Usage:
        To create a new trading strategy, subclass this Strategy class and
        implement the required abstract methods, particularly the 'on_bar' method
        which defines the core logic of the strategy. Other methods can be
        overridden as needed to customize behavior.

        Example:
        ```
        class MyStrategy(Strategy):
            def __init__(self, name, parameters):
                super().__init__(name, parameters)

            def on_bar(self, bar):
                # Implement strategy logic here
                pass

            def _handle_order_update(self, order):
                # Custom order handling logic
                pass

            def _handle_trade_update(self, trade):
                # Custom trade handling logic
                pass
        ```

    Note:
        This class is designed to be used within a larger backtesting or trading
        system. It requires integration with a data feed, order execution system,
        and portfolio management component to function fully.
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
            parameters (Optional[Dict[str, Any]]): Initial strategy parameters. Defaults to None.
            primary_timeframe (Optional[Timeframe]): The primary timeframe for the strategy. Defaults to None.
        """
        self.name: str = name
        self._parameters: Parameters = Parameters(parameters or {})
        self.primary_timeframe: Optional[Timeframe] = primary_timeframe
        self._datas: Dict[str, Dict[Timeframe, Data]] = defaultdict(dict)
        self._bar_manager: BarManager = BarManager()
        self._initialized: bool = False
        self._primary_symbol: Optional[str] = None
        self._engine = None
        self._strategy_timeframes: Dict[Timeframe, List[str]] = {}
        self._id: str = generate_unique_id()
        self._positions: Dict[str, float] = {}
        self._pending_orders: List[Order] = []
        self._open_trades: Dict[str, List[Trade]] = defaultdict(list)
        self._closed_trades: List[Trade] = []

    # region Initialization and Configuration

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
            logger_main.log_and_raise(StrategyError("Strategy is already initialized."))

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
            self._positions[symbol] = 0.0

        self._primary_symbol = symbols[0] if symbols else None
        self._initialized = True
        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self.primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

    # endregion

    # region Data Management

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
            logger_main.log_and_raise(
                StrategyError("Strategy has not been initialized.")
            )
        if self._primary_symbol is None:
            logger_main.log_and_raise(
                StrategyError("No primary symbol set for the strategy.")
            )
        try:
            return self._datas[self._primary_symbol][self.primary_timeframe]
        except KeyError:
            logger_main.log_and_raise(
                StrategyError(
                    f"Data for primary symbol {self._primary_symbol} and timeframe {self.primary_timeframe} not found."
                )
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

    def _process_bar(self, bar: Bar) -> None:
        """
        Process a new bar and update the strategy's state.

        This method is called by the Engine for each new bar. It updates the strategy's
        internal data structures and calls the on_bar method for decision making.

        Args:
            bar (Bar): The new price bar data.
        """
        symbol, timeframe = bar.ticker, bar.timeframe
        self._datas[symbol][timeframe].add_bar(bar)
        self._bar_manager.add_bar(bar)
        self.on_bar(bar)

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
        Create a buy order.

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
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        parent_order, child_orders = self._engine.create_order(
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

        if parent_order:
            self._pending_orders.append(parent_order)
            self._pending_orders.extend(child_orders)

        return parent_order, child_orders

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
        Create a sell order.

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
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        parent_order, child_orders = self._engine.create_order(
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

        if parent_order:
            self._pending_orders.append(parent_order)
            self._pending_orders.extend(child_orders)

        return parent_order, child_orders

    def add_stop_loss(self, symbol: str, stop_loss: float) -> Optional[Order]:
        """
        Add a stop-loss order to an existing position.

        Args:
            symbol (str): The symbol for the position.
            stop_loss (float): The stop-loss price.

        Returns:
            Optional[Order]: The created stop-loss Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if there's no open position.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        position = self.get_current_position(symbol)
        if position == 0:
            logger_main.log_and_raise(StrategyError(f"No open position for {symbol}"))

        direction = Order.Direction.SHORT if position > 0 else Order.Direction.LONG
        order = self._engine.create_order(
            self._id, symbol, direction, abs(position), Order.ExecType.STOP, stop_loss
        )[0]  # We only need the parent order here

        if order:
            self._pending_orders.append(order)

        return order

    def add_take_profit(self, symbol: str, take_profit: float) -> Optional[Order]:
        """
        Add a take-profit order to an existing position.

        Args:
            symbol (str): The symbol for the position.
            take_profit (float): The take-profit price.

        Returns:
            Optional[Order]: The created take-profit Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if there's no open position.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        position = self.get_current_position(symbol)
        if position == 0:
            logger_main.log_and_raise(StrategyError(f"No open position for {symbol}"))

        direction = Order.Direction.SHORT if position > 0 else Order.Direction.LONG
        order = self._engine.create_order(
            self._id,
            symbol,
            direction,
            abs(position),
            Order.ExecType.LIMIT,
            take_profit,
        )[0]  # We only need the parent order here

        if order:
            self._pending_orders.append(order)

        return order

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
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )
        success = self._engine.cancel_order(self._id, order)
        if success:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
        return success

    def close(self, symbol: Optional[str] = None, size: Optional[float] = None) -> bool:
        """
        Close all positions for this strategy, or for a specific symbol if provided.
        If size is specified, it performs a partial close.

        Args:
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.
            size (Optional[float]): The size of the position to close. If None, close the entire position.

        Returns:
            bool: True if the closing operation was successful, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        if size is not None:
            if symbol is None:
                logger_main.log_and_raise(
                    StrategyError("Symbol must be specified for partial close.")
                )
            return self._partial_close(symbol, size) is not None

        success = self._engine.close_positions(self._id, symbol)
        if success and symbol:
            self._positions[symbol] = 0.0
        elif success:
            self._positions = {s: 0.0 for s in self._positions}
        return success

    def _partial_close(self, symbol: str, size: float) -> Optional[Order]:
        """
        Partially close an existing position.

        Args:
            symbol (str): The symbol for the position.
            size (float): The size of the position to close.

        Returns:
            Optional[Order]: The created Order object for the partial close, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if there's no open position.
        """
        position = self.get_current_position(symbol)
        if position == 0:
            logger_main.log_and_raise(StrategyError(f"No open position for {symbol}"))

        if abs(size) > abs(position):
            logger_main.log_and_raise(
                StrategyError(
                    f"Requested size {size} is larger than current position {position}"
                )
            )

        direction = Order.Direction.SHORT if position > 0 else Order.Direction.LONG
        order = self._engine.create_order(
            self._id, symbol, direction, abs(size), Order.ExecType.MARKET
        )[0]  # We only need the parent order here

        if order:
            self._pending_orders.append(order)

        return order

    # endregion

    # region Position Management

    def get_current_position(self, symbol: str) -> float:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            float: The current position size (positive for long, negative for short, 0 for no position).
        """
        return self._engine.get_position_size(symbol)

    def calculate_position_size(
        self, fill_price: float, risk_percent: float, stop_loss: float
    ) -> float:
        """
        Calculate the position size based on fill_price, risk percentage and stop loss.

        Args:
            fill_price (float): The fill price.
            risk_percent (float): The percentage of account to risk on this trade.
            stop_loss (float): The stop loss price.

        Returns:
            float: The calculated position size.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if the risk per share is zero.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        account_value = self._engine.get_account_value()
        risk_amount = account_value * (risk_percent / 100)
        risk_per_share = abs(fill_price - stop_loss)

        if risk_per_share == 0:
            logger_main.log_and_raise(
                StrategyError("Risk per share is zero. Cannot calculate position size.")
            )

        available_margin = self._engine.get_available_margin()
        max_position_size = available_margin / fill_price

        position_size = min(risk_amount / risk_per_share, max_position_size)
        return position_size

    def enforce_risk_limits(self, symbol: str, proposed_position_size: float) -> float:
        """
        Enforce risk limits on the proposed position size.

        Args:
            symbol (str): The symbol to trade.
            proposed_position_size (float): The initially calculated position size.

        Returns:
            float: The adjusted position size that complies with risk limits.
        """
        # Example risk limit: no single position can be more than 5% of the account value
        account_value = self._engine.get_account_value()
        current_price = self._datas[symbol][self.primary_timeframe].close[-1]
        max_position_value = account_value * 0.05
        max_position_size = max_position_value / current_price

        return min(proposed_position_size, max_position_size)

    # endregion

    # region Order and Trade Update Handling

    def _on_order_update(self, order: Order) -> None:
        """
        Handle updates to orders created by this strategy.

        Args:
            order (Order): The updated Order object.
        """
        if order.status == Order.Status.FILLED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
            self._update_position(order)
        elif order.status == Order.Status.PARTIALLY_FILLED:
            self._handle_partial_fill(order)
        elif order.status == Order.Status.CANCELED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]

        self.on_order_update(order)

    def _on_trade_update(self, trade: Trade) -> None:
        """
        Handle updates to trades associated with this strategy.

        Args:
            trade (Trade): The updated Trade object.
        """
        if trade.status == Trade.Status.CLOSED:
            self._update_position(trade)
            self._open_trades[trade.ticker] = [
                t for t in self._open_trades[trade.ticker] if t.id != trade.id
            ]
            self._closed_trades.append(trade)
        elif trade.status == Trade.Status.PARTIALLY_CLOSED:
            self._update_position(trade)

        self.on_trade_update(trade)

    def _handle_partial_fill(self, order: Order) -> None:
        """
        Handle a partial fill of an order.

        Args:
            order (Order): The partially filled Order object.
        """
        # Update the strategy's state based on the partial fill
        filled_size = order.get_filled_size()
        remaining_size = order.get_remaining_size()

        # Update the order in pending orders
        for i, pending_order in enumerate(self._pending_orders):
            if pending_order.id == order.id:
                self._pending_orders[i] = order
                break

        # Implement strategy-specific logic for handling partial fills
        self.on_partial_fill(order, filled_size, remaining_size)

    def _update_position(self, transaction: Union[Order, Trade]) -> None:
        """
        Update the strategy's position based on an order or trade.

        Args:
            transaction (Union[Order, Trade]): The Order or Trade object to update the position from.
        """
        symbol = transaction.ticker
        if isinstance(transaction, Order):
            size = transaction.details.size
        else:  # Trade
            size = transaction.current_size

        if transaction.direction == Order.Direction.LONG:
            self._positions[symbol] += size
        else:  # SHORT
            self._positions[symbol] -= size

    # endregion

    # region Limit Exit Order Management

    def manage_stop_loss(self, symbol: str, new_stop_loss: float) -> bool:
        """
        Adjust the stop-loss level for an existing position.

        Args:
            symbol (str): The symbol for the position.
            new_stop_loss (float): The new stop-loss price.

        Returns:
            bool: True if the stop-loss was successfully adjusted, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if there's no open position.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        position = self.get_current_position(symbol)
        if position == 0:
            logger_main.log_and_raise(StrategyError(f"No open position for {symbol}"))

        for order in self._pending_orders:
            if (
                order.details.ticker == symbol
                and order.details.exectype == Order.ExecType.STOP
                and (
                    (position > 0 and order.details.direction == Order.Direction.SHORT)
                    or (
                        position < 0 and order.details.direction == Order.Direction.LONG
                    )
                )
            ):
                return self._engine.modify_order(order.id, {"price": new_stop_loss})

        # If no existing stop-loss order was found, create a new one
        return self.add_stop_loss(symbol, new_stop_loss) is not None

    def manage_take_profit(self, symbol: str, new_take_profit: float) -> bool:
        """
        Adjust the take-profit level for an existing position.

        Args:
            symbol (str): The symbol for the position.
            new_take_profit (float): The new take-profit price.

        Returns:
            bool: True if the take-profit was successfully adjusted, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if there's no open position.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        position = self.get_current_position(symbol)
        if position == 0:
            logger_main.log_and_raise(StrategyError(f"No open position for {symbol}"))

        for order in self._pending_orders:
            if (
                order.details.ticker == symbol
                and order.details.exectype == Order.ExecType.LIMIT
                and (
                    (position > 0 and order.details.direction == Order.Direction.SHORT)
                    or (
                        position < 0 and order.details.direction == Order.Direction.LONG
                    )
                )
            ):
                return self._engine.modify_order(order.id, {"price": new_take_profit})

        # If no existing take-profit order was found, create a new one
        return self.add_take_profit(symbol, new_take_profit) is not None

    # endregion

    # region Performance Tracking

    def calculate_strategy_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for the strategy.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.
        """
        if self._engine is None:
            logger_main.log_and_raise(
                StrategyError("Strategy is not connected to an engine.")
            )

        open_trades = self._engine.get_open_trades(self._id)
        closed_trades = self._engine.get_closed_trades(self._id)

        total_trades = len(open_trades) + len(closed_trades)
        winning_trades = sum(1 for trade in closed_trades if trade.metrics.pnl > 0)
        losing_trades = sum(1 for trade in closed_trades if trade.metrics.pnl < 0)

        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        avg_win = (
            sum(trade.metrics.pnl for trade in closed_trades if trade.metrics.pnl > 0)
            / winning_trades
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(
                abs(trade.metrics.pnl)
                for trade in closed_trades
                if trade.metrics.pnl < 0
            )
            / losing_trades
            if losing_trades
            else 0
        )
        profit_factor = avg_win / avg_loss if avg_loss != 0 else float("inf")

        return {
            "total_trades": total_trades,
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            # Add more metrics as needed
        }

    # endregion

    # region Abstract Methods

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """
        Handle the arrival of a new price bar.

        This method is the primary decision-making point for the strategy.
        It should be implemented by concrete strategy classes to define the strategy's logic.

        Args:
            bar (Bar): The new price bar data.
        """
        pass

    def on_partial_fill(
        self, order: Order, filled_size: float, remaining_size: float
    ) -> None:
        """
        Handle strategy-specific logic for partial fills.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to partial fills.

        Args:
            order (Order): The partially filled Order object.
            filled_size (float): The size that was filled.
            remaining_size (float): The remaining size to be filled.
        """
        pass

    def on_order_update(self, order: Order) -> None:
        """
        Handle strategy-specific logic for order updates.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to order updates.

        Args:
            order (Order): The updated Order object.
        """
        pass

    def on_trade_update(self, trade: Trade) -> None:
        """
        Handle strategy-specific logic for trade updates.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to trade updates.

        Args:
            trade (Trade): The updated Trade object.
        """
        pass

    # endregion

    # region Utility Methods

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
            logger_main.log_and_raise(KeyError(f"Invalid key: {key}"))

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

    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders for this strategy.

        Returns:
            List[Order]: A list of all pending orders.
        """
        return self._pending_orders

    def get_positions(self) -> Dict[str, float]:
        """
        Get all current positions for this strategy.

        Returns:
            Dict[str, float]: A dictionary of current positions, keyed by symbol.
        """
        return self._positions.copy()

    def set_engine(self, engine: Any) -> None:
        """
        Set the engine for this strategy.

        Args:
            engine (Any): The engine object to set.
        """
        self._engine = engine

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return (
            f"Strategy(name={self.name}, id={self._id}, "
            f"symbols={list(self._datas.keys())}, "
            f"timeframes={list(next(iter(self._datas.values())).keys())}, "
            f"positions={self._positions})"
        )

    # endregion
