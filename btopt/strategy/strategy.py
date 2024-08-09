import uuid
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..data.timeframe import Timeframe
from ..indicator.indicator import Indicator
from ..log_config import logger_main
from ..order import Order
from ..parameters import Parameters
from ..trade import Trade
from ..util.metaclasses import PreInitABCMeta
from .helper import Data, DataTimeframe


class StrategyError(Exception):
    pass


def generate_unique_id() -> str:
    """Generate a unique identifier for the strategy."""
    return str(uuid.uuid4())


class Strategy(metaclass=PreInitABCMeta):
    def __init__(
        self,
        name: Optional[str] = None,
        primary_timeframe: Optional[Timeframe] = None,
        parameters: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        self._id: str = generate_unique_id()
        self.name: str = name or self._id
        self.datas: Dict[str, Data] = {}

        self._engine: Any = None
        self._indicators: Dict[str, Indicator] = {}
        self._initialized: bool = False
        self._parameters: Parameters = Parameters(parameters or {})
        self._primary_symbol: Optional[str] = None
        self._primary_timeframe: Optional[Timeframe] = primary_timeframe
        self._strategy_timeframes: List[Timeframe] = []

        self._pending_orders: List[Order] = []
        self._positions: Dict[str, float] = {}
        self._open_trades: Dict[str, List[Trade]] = defaultdict(list)
        self._closed_trades: List[Trade] = []

        self._max_bars_back: int = 500
        self._warmup_period: int = 1
        self._is_warmup_complete = False

        self._base_init_called = True

    # region Initialization and Configuration

    def initialize(
        self,
        symbols: List[str],
        timeframes: List[Timeframe],
    ) -> None:
        if self._initialized:
            logger_main.log_and_raise(StrategyError("Strategy is already initialized."))

        default_timeframe = min(timeframes)

        if self._primary_timeframe is None:
            self._primary_timeframe = default_timeframe
        elif self._primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self._primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self._primary_timeframe = default_timeframe

        # Create Data objects for each symbol
        for symbol in symbols:
            self.datas[symbol] = Data(symbol)

        # Set the strategy timeframes
        self._strategy_timeframes = timeframes
        self._primary_symbol = symbols[0] if symbols else None
        self._initialized = True

        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self._primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

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
            StrategyError: If the strategy hasn't been initialized or if there's no primary symbol set.
        """
        if not self._initialized:
            logger_main.log_and_raise(
                StrategyError("Strategy has not been initialized.")
            )
        if self._primary_symbol is None:
            logger_main.log_and_raise(
                StrategyError("No primary symbol set for the strategy.")
            )
        return self.datas[self._primary_symbol][self._primary_timeframe]

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
    def max_bars_back(self) -> int:
        """
        Get the maximum number of bars to keep in history.

        Returns:
            int: The maximum number of bars.
        """
        return self._max_bars_back

    @max_bars_back.setter
    def max_bars_back(self, value: int) -> None:
        """
        Set the maximum number of bars to keep in history and update related components.

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

        # Update all Data objects
        for data in self.datas.values():
            data.max_length = value

        logger_main.info(f"Updated max_bars_back to {value}")

    @property
    def warmup_period(self) -> int:
        """
        Get the warmup period for the strategy.

        Returns:
            int: The current warmup period.
        """
        return self._warmup_period

    @warmup_period.setter
    def warmup_period(self, value: int) -> None:
        """
        Set the warmup period for the strategy.

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

    def _check_warmup_period(self) -> bool:
        """
        Check if all timeframes for all symbols have at least warmup_period data points.

        Returns:
            bool: True if all timeframes have sufficient data, False otherwise.
        """
        if self._is_warmup_complete:
            return True

        for symbol, data in self.datas.items():
            for timeframe in self._strategy_timeframes:
                if timeframe not in data.timeframes:
                    logger_main.warning(
                        f"Timeframe `{repr(timeframe)}` was not found for symbol `{symbol}`. Available timeframes are {self._strategy_timeframes}"
                    )
                    return False

                elif len(data[timeframe]) < self.warmup_period:
                    return False

        self._is_warmup_complete = True
        logger_main.info(f"Strategy {self._id} warm up is complete.")
        return True

    def get_data_timeframe(self, symbol: str, timeframe: Timeframe) -> DataTimeframe:
        """
        Get a DataTimeframe object for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.

        Returns:
            DataTimeframe: The DataTimeframe object for the specified symbol and timeframe.

        Raises:
            KeyError: If the requested symbol is not found.
        """
        if symbol not in self.datas:
            raise KeyError(f"Symbol {symbol} not found in strategy data.")
        return self.datas[symbol][timeframe]

    def _on_data(self):
        # Update indicators
        # for indicator in self._indicators.values():
        #     indicator.on_data()

        if self._check_warmup_period():
            self.on_data()

    # endregion

    # region Indicator Management
    def add_indicator(
        self, indicator: Type[Indicator], name: str, **kwargs: Any
    ) -> None:
        """
        Add an indicator to the strategy.

        This method creates an instance of the specified indicator and associates it with the strategy.

        Args:
            indicator (Type[Indicator]): The indicator class to instantiate.
            name (str): A unique name for the indicator instance.
            **kwargs: Additional keyword arguments to pass to the indicator constructor.

        Raises:
            ValueError: If an indicator with the same name already exists.
        """
        if name in self._indicators:
            logger_main.log_and_raise(
                ValueError(
                    f"An indicator named '{name}' already exists in this strategy"
                )
            )

        if self._primary_symbol is None:
            logger_main.log_and_raise(
                ValueError("No primary symbol set for the strategy")
            )

        indicator_instance = indicator(
            self.get_data(self._primary_symbol), name=name, **kwargs
        )

        # Align indicator timeframe with strategy timeframe
        indicator_instance._align_timeframes(self._primary_timeframe)

        self._indicators[name] = indicator_instance
        logger_main.info(f"Added indicator '{name}' to strategy '{self.name}'")

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

        return self._indicators[indicator_name].get_output(output_name)

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
        current_price = self.get_data_timeframe(symbol, self._primary_timeframe).close[
            -1
        ]
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
            self._positions[symbol] = self._positions.get(symbol, 0) + size
        else:  # SHORT
            self._positions[symbol] = self._positions.get(symbol, 0) - size

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
        }

    # endregion

    # region Abstract Methods

    @abstractmethod
    def on_data(self) -> None:
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
    def universe_selection(self):
        """
        NotImplemented: Filter and sort self.datas based on a rule
        """
        pass

    def get_data(self, symbol: Optional[str] = None) -> Data:
        """
        Get the Data object for a specific symbol.

        This method allows access to Data objects for symbols other than the primary symbol.

        Args:
            symbol (str): The symbol for which to retrieve the Data object.

        Returns:
            Data: The Data object for the specified symbol.

        Raises:
            StrategyError: If the strategy hasn't been initialized or if the symbol doesn't exist.
        """
        if not self._initialized:
            logger_main.log_and_raise(
                StrategyError("Strategy has not been initialized.")
            )

        symbol = symbol or self._primary_symbol
        if symbol not in self.datas:
            logger_main.log_and_raise(
                StrategyError(f"No data available for symbol: {symbol}")
            )
        return self.datas[symbol]

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

    def log(self, message: str):
        timestamp = ""
        if self._engine:
            timestamp = f"\n\nDatetime: {self._engine._current_timestamp} | "

        logger_main.warning(timestamp + message)

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return (
            f"Strategy(name={self.name}, id={self._id}, "
            f"symbols={list(self.datas.keys())}, "
            f"primary_timeframe={self._primary_timeframe}, "
            f"positions={self._positions})"
        )

    def __len__(self) -> int:
        """
        Get the length of stored data for the primary symbol and timeframe.

        Returns:
            int: The number of available data points for the primary symbol and timeframe.

        Raises:
            StrategyError: If the strategy hasn't been initialized or if there's no primary symbol set.
        """
        if not self._initialized:
            logger_main.log_and_raise(
                StrategyError("Strategy has not been initialized.")
            )

        if self._primary_symbol is None:
            logger_main.log_and_raise(
                StrategyError("No primary symbol set for the strategy.")
            )

        return self.get_data_length(self._primary_symbol, self._primary_timeframe)

    # endregion
