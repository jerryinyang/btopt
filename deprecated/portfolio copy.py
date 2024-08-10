from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .data.bar import Bar
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails
from .trade import Trade
from .types import EngineType
from .util.decimal import ExtendedDecimal


class Portfolio:
    """
    A comprehensive portfolio management system for trading and backtesting.

    The Portfolio class represents a trading account and provides functionality for
    order management, trade execution, position tracking, risk management, and
    performance analysis. It supports multiple assets, various order types, and
    margin trading.

    Key Features:
    - Order management: Create, execute, modify, and cancel orders
    - Position tracking: Monitor open positions and average entry prices
    - Trade management: Open, close, and partially close trades
    - Risk management: Implement margin trading with configurable ratios and margin calls
    - Performance analysis: Calculate key metrics such as total return, Sharpe ratio, and drawdown
    - Multi-strategy support: Track trades and orders for multiple trading strategies
    - Reporting: Generate comprehensive reports and visualizations using the integrated Reporter class

    Attributes:
        initial_capital (ExtendedDecimal): The starting capital of the portfolio.
        cash (ExtendedDecimal): The current cash balance.
        commission_rate (ExtendedDecimal): The commission rate for trades.
        slippage (ExtendedDecimal): The slippage rate for trades.
        pyramiding (int): The maximum number of allowed positions per symbol.
        margin_ratio (ExtendedDecimal): The required margin ratio for trades.
        margin_call_threshold (ExtendedDecimal): The threshold for triggering a margin call.
        positions (Dict[str, ExtendedDecimal]): Current positions for each symbol.
        avg_entry_prices (Dict[str, ExtendedDecimal]): Average entry prices for each symbol.
        open_trades (Dict[str, List[Trade]]): Open trades grouped by symbol.
        closed_trades (List[Trade]): List of closed trades.
        pending_orders (List[Order]): List of pending orders.
        limit_exit_orders (List[Order]): List of pending limit exit orders.
        trade_count (int): Total number of trades executed.
        margin_used (ExtendedDecimal): Amount of margin currently in use.
        buying_power (ExtendedDecimal): Available buying power for new trades.
        metrics (pd.DataFrame): DataFrame storing portfolio metrics over time.
        engine (Engine): Instance of the Engine class managing the trading system.

    Usage:
        portfolio = Portfolio(initial_capital=ExtendedDecimal("100000"), commission_rate=ExtendedDecimal("0.001"))
        portfolio.create_order("AAPL", Order.Direction.LONG, 100, Order.ExecType.MARKET)
        portfolio.update(timestamp, market_data)
        performance = portfolio.get_performance_metrics()
    """

    def __init__(
        self,
        initial_capital: ExtendedDecimal = ExtendedDecimal("100000"),
        commission_rate: ExtendedDecimal = ExtendedDecimal("0.001"),
        slippage: ExtendedDecimal = ExtendedDecimal("0"),
        pyramiding: int = 1,
        margin_ratio: ExtendedDecimal = ExtendedDecimal("0.5"),
        margin_call_threshold: ExtendedDecimal = ExtendedDecimal("0.3"),
        engine: EngineType = None,
    ):
        """
        Initialize the Portfolio with given parameters.

        Args:
            initial_capital (ExtendedDecimal): The starting capital of the portfolio.
            commission_rate (ExtendedDecimal): The commission rate for trades.
            slippage (ExtendedDecimal): The slippage rate for trades.
            pyramiding (int): The maximum number of allowed positions per symbol.
            margin_ratio (ExtendedDecimal): The required margin ratio for trades.
            margin_call_threshold (ExtendedDecimal): The threshold for triggering a margin call.
            engine (Optional['Engine']): The Engine instance managing the trading system.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.pyramiding = pyramiding
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.engine = engine

        self.positions: Dict[str, Dict[str, ExtendedDecimal]] = {}
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.limit_exit_orders: List[Order] = []
        self.trade_count = 0
        self.margin_used = ExtendedDecimal("0")
        self.buying_power = initial_capital / margin_ratio

        self.updated_orders: List[Order] = []
        self.updated_trades: List[Trade] = []

        self.metrics = pd.DataFrame(
            columns=[
                "timestamp",
                "cash",
                "equity",
                "open_pnl",
                "closed_pnl",
                "portfolio_return",
            ]
        )

    # region Initialization and Configuration

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the portfolio.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.initial_capital = ExtendedDecimal(
            str(config.get("initial_capital", self.initial_capital))
        )
        self.commission_rate = ExtendedDecimal(
            str(config.get("commission_rate", self.commission_rate))
        )
        self.slippage = ExtendedDecimal(str(config.get("slippage", self.slippage)))
        self.pyramiding = config.get("pyramiding", self.pyramiding)
        self.margin_ratio = ExtendedDecimal(
            str(config.get("margin_ratio", self.margin_ratio))
        )
        self.margin_call_threshold = ExtendedDecimal(
            str(config.get("margin_call_threshold", self.margin_call_threshold))
        )

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.
        """
        self.__init__(
            self.initial_capital,
            self.commission_rate,
            self.slippage,
            self.pyramiding,
            self.margin_ratio,
            self.margin_call_threshold,
            self.engine,
        )
        logger_main.info("Portfolio reset to initial state.")

    # endregion

    # region Portfolio Update and Market Data Processing

    def update(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the portfolio state based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        self._process_pending_orders(timestamp, market_data)
        self._update_open_trades(market_data)
        self._update_metrics(timestamp, market_data)

    def _update_metrics(
        self, timestamp: pd.Timestamp, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the portfolio metrics based on the current market data.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        self.update_position_values(market_data)

        equity = self.calculate_equity()
        open_pnl = sum(
            trade.metrics.unrealized_pnl
            for trades in self.open_trades.values()
            for trade in trades
        )
        closed_pnl = sum(trade.metrics.realized_pnl for trade in self.closed_trades)

        if not self.metrics.empty:
            previous_equity = self.metrics.iloc[-1]["equity"]
            portfolio_return = (equity - previous_equity) / previous_equity
        else:
            portfolio_return = ExtendedDecimal("0")

        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "cash": [self.cash],
                "equity": [equity],
                "open_pnl": [open_pnl],
                "closed_pnl": [closed_pnl],
                "portfolio_return": [portfolio_return],
            }
        )

        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

    def _get_current_price(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current price for a symbol.

        Args:
            symbol (str): The symbol to get the price for.

        Returns:
            ExtendedDecimal: The current price of the symbol.

        Raises:
            ValueError: If unable to get the current price for the symbol.
        """
        if self.engine and hasattr(self.engine, "_current_market_data"):
            market_data = self.engine._current_market_data[symbol]
            current_data = market_data[min(market_data.keys())]
            if current_data is not None:
                return ExtendedDecimal(str(current_data.close))

        logger_main.log_and_raise(
            ValueError(f"Unable to get current price for {symbol}")
        )

    # endregion

    # region Order and Trade Management

    def create_order(
        self,
        symbol: str,
        direction: Order.Direction,
        size: float,
        order_type: Order.ExecType,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Order:
        """
        Create and add a new order to pending orders.

        Args:
            symbol (str): The symbol for the order.
            direction (Order.Direction): The direction of the order (LONG or SHORT).
            size (float): The size of the order.
            order_type (Order.ExecType): The type of the order (e.g., MARKET, LIMIT).
            price (Optional[float]): The price for limit orders.
            **kwargs: Additional order parameters.

        Returns:
            Order: The created order object.
        """
        order_details = OrderDetails(
            ticker=symbol,
            direction=direction,
            size=ExtendedDecimal(str(size)),
            price=ExtendedDecimal(str(price)) if price is not None else None,
            exectype=order_type,
            timestamp=datetime.now(),
            **kwargs,
        )
        order = Order(order_id=self._generate_order_id(), details=order_details)
        self.add_pending_order(order)
        return order

    def execute_order(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update the portfolio accordingly.

        Args:
            order (Order): The order to execute.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing a boolean indicating if the order was executed
                                        successfully, and the resulting Trade object if applicable.
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        cost = execution_price * size
        commission = cost * self.commission_rate

        if not self._check_margin_requirements(order, cost):
            logger_main.info(f"Insufficient margin to execute order: {order}")
            return False, None

        self._update_margin(order, cost)

        # Update cash based on the order direction
        if direction == Order.Direction.LONG:
            self.cash -= cost + commission
        else:  # SHORT
            self.cash += cost - commission

        # Update or create position
        self._update_position(symbol, size * direction.value, execution_price)

        # Create or update trade
        if symbol not in self.open_trades:
            self.open_trades[symbol] = []

        self.trade_count += 1
        new_trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
        )
        new_trade.initial_size = size
        new_trade.current_size = size
        self.open_trades[symbol].append(new_trade)

        # Update order status
        order.fill(execution_price, bar.timestamp, size)

        self.updated_orders.append(order)
        self.updated_trades.append(new_trade)

        logger_main.info(f"Executed order: {order}, resulting trade: {new_trade}")
        return True, new_trade

    def add_pending_order(self, order: Order) -> None:
        """
        Add an order to the appropriate list of pending orders.

        Args:
            order (Order): The order to add to pending orders.
        """
        if (
            order.family_role == Order.FamilyRole.CHILD_EXIT
            and order.details.exectype
            in [Order.ExecType.EXIT_LIMIT, Order.ExecType.EXIT_STOP]
        ):
            self.limit_exit_orders.append(order)
        else:
            self.pending_orders.append(order)
        logger_main.info(f"Added pending order: {order}")

    def cancel_order(self, order: Order) -> bool:
        """
        Cancel an order and update its status.

        Args:
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        elif order in self.limit_exit_orders:
            self.limit_exit_orders.remove(order)
        else:
            logger_main.warning(
                f"Failed to cancel order (not found in pending orders): {order}",
            )
            return False

        order.status = Order.Status.CANCELED
        self.updated_orders.append(order)
        logger_main.info(f"Cancelled order: {order}")
        return True

    def modify_order(self, order_id: int, new_details: Dict[str, Any]) -> bool:
        """
        Modify an existing pending order.

        Args:
            order_id (int): The ID of the order to modify.
            new_details (Dict[str, Any]): A dictionary containing the new details for the order.

        Returns:
            bool: True if the order was successfully modified, False otherwise.
        """
        for order in self.pending_orders:
            if order.id == order_id:
                for key, value in new_details.items():
                    if hasattr(order.details, key):
                        setattr(order.details, key, value)
                self.updated_orders.append(order)
                logger_main.info(f"Modified order: {order}")
                return True
        logger_main.info(f"Order with ID {order_id} not found in pending orders.")
        return False

    def update_position_values(
        self, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the current market value of all open positions.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols.
        """
        for symbol, position in self.positions.items():
            if symbol in market_data:
                timeframe = min(market_data[symbol].keys())
                current_price = market_data[symbol][timeframe].close
                position["current_price"] = current_price
                position["market_value"] = position["size"] * current_price

    def close_positions(self, strategy_id: str, symbol: Optional[str] = None) -> bool:
        """
        Close positions for a specific strategy and/or symbol.

        Args:
            strategy_id (str): The ID of the strategy.
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.

        Returns:
            bool: True if any positions were closed, False otherwise.
        """
        closed_any = False
        trades_to_close = self._get_open_trades(strategy_id, symbol)

        for trade in trades_to_close:
            self.close_trade(trade, trade.current_price)
            closed_any = True

        if closed_any:
            logger_main.info(
                f"Closed positions for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
            )
        else:
            logger_main.info(
                f"No positions to close for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
            )

        return closed_any

    def close_all_positions(self, current_timestamp: datetime) -> None:
        """
        Close all open positions in the portfolio.

        Args:
            current_timestamp (datetime): The current timestamp to use for closing trades.
        """
        symbols_to_close = list(self.open_trades.keys())

        for symbol in symbols_to_close:
            trades_to_close = self.open_trades[symbol][:]  # Create a copy of the list
            for trade in trades_to_close:
                current_price = self._get_current_price(symbol)
                self.close_trade(trade, current_price)

        logger_main.info("Closed all open positions.")

    def close_trade(self, trade: Trade, current_price: ExtendedDecimal) -> None:
        """
        Close a specific trade.

        Args:
            trade (Trade): The trade to close.
            current_price (ExtendedDecimal): The current market price to close the trade at.
        """
        symbol = trade.ticker
        close_size = trade.current_size
        close_value = close_size * current_price
        commission = close_value * self.commission_rate

        # Update cash
        if trade.direction == Order.Direction.LONG:
            self.cash += close_value - commission
        else:  # SHORT
            self.cash -= close_value + commission

        # Update position
        self._update_position(
            symbol, -close_size * trade.direction.value, current_price
        )

        # Close the trade
        trade.close(None, current_price)

        # Move trade from open to closed
        if symbol in self.open_trades:
            self.open_trades[symbol].remove(trade)
            if not self.open_trades[symbol]:
                del self.open_trades[symbol]
        self.closed_trades.append(trade)

        self.updated_trades.append(trade)

        logger_main.info(f"Closed trade: {trade}")

    def _get_open_trades(
        self, strategy_id: str, symbol: Optional[str] = None
    ) -> List[Trade]:
        """
        Get open trades for a specific strategy and/or symbol.

        Args:
            strategy_id (str): The ID of the strategy.
            symbol (Optional[str]): The symbol to filter trades for. If None, return all open trades.

        Returns:
            List[Trade]: A list of open trades matching the criteria.
        """
        trades = []
        for sym, trade_list in self.open_trades.items():
            if symbol is None or sym == symbol:
                trades.extend([t for t in trade_list if t.strategy_id == strategy_id])
        return trades

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        orders_to_process: List[Order] = (
            self.pending_orders[:] + self.limit_exit_orders[:]
        )

        for order in orders_to_process:
            symbol = order.details.ticker
            timeframe = order.details.timeframe

            # Handle case where order timeframe is None
            if timeframe is None:
                available_timeframes = list(market_data[symbol].keys())
                if not available_timeframes:
                    logger_main.warning(
                        f"No market data available for symbol {symbol}. Skipping order processing.",
                    )
                    continue
                timeframe = min(available_timeframes)
                logger_main.warning(
                    f"Order for {symbol} has no timeframe. Using lowest available: {timeframe}",
                )

            try:
                current_bar = market_data[symbol][timeframe]
            except KeyError:
                logger_main.warning(
                    f"No market data for {symbol} at timeframe {timeframe}. Skipping order.",
                )
                continue

            is_filled, fill_price = order.is_filled(current_bar)
            if is_filled:
                executed, trade = self.execute_order(order, fill_price, current_bar)
                if executed:
                    self._remove_executed_order(order)
                    self.updated_orders.append(order)
                    if trade:
                        self.updated_trades.append(trade)
            elif order.is_expired(timestamp):
                self._cancel_expired_order(order)

        # Check for margin call after processing orders
        if self._check_margin_call():
            self._handle_margin_call()

    def _remove_executed_order(self, order: Order) -> None:
        """Remove an executed order from the appropriate list."""
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        elif order in self.limit_exit_orders:
            self.limit_exit_orders.remove(order)

    def _cancel_expired_order(self, order: Order) -> None:
        """Cancel an expired order and update its status."""
        self._remove_executed_order(order)
        order.status = Order.Status.CANCELED
        self.updated_orders.append(order)
        logger_main.info(
            f"Order {order.id} for {order.details.ticker} has expired and been canceled.",
        )

    def _update_position(
        self, symbol: str, size_change: ExtendedDecimal, price: ExtendedDecimal
    ) -> None:
        """
        Update the position for a given symbol.

        Args:
            symbol (str): The symbol of the position to update.
            size_change (ExtendedDecimal): The change in position size (positive for buy, negative for sell).
            price (ExtendedDecimal): The price at which the position is being updated.
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "size": ExtendedDecimal("0"),
                "avg_price": ExtendedDecimal("0"),
            }

        current_size = self.positions[symbol]["size"]
        current_avg_price = self.positions[symbol]["avg_price"]

        new_size = current_size + size_change

        if new_size == ExtendedDecimal("0"):
            del self.positions[symbol]
        else:
            if current_size == ExtendedDecimal("0"):
                new_avg_price = price
            else:
                if (
                    current_size > ExtendedDecimal("0")
                    and size_change > ExtendedDecimal("0")
                ) or (
                    current_size < ExtendedDecimal("0")
                    and size_change < ExtendedDecimal("0")
                ):
                    # Increasing position
                    new_avg_price = (
                        current_size * current_avg_price + size_change * price
                    ) / new_size
                else:
                    # Reducing position
                    new_avg_price = current_avg_price

            self.positions[symbol] = {"size": new_size, "avg_price": new_avg_price}

    def _update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        for symbol, trades in self.open_trades.items():
            timeframe = min(market_data[symbol].keys())
            current_bar = market_data[symbol][timeframe]
            for trade in trades:
                trade.update(current_bar)
                self.updated_trades.append(trade)

    def _create_or_update_trade(
        self,
        order: Order,
        execution_price: ExtendedDecimal,
        bar: Bar,
        size: Optional[ExtendedDecimal] = None,
    ) -> Trade:
        """
        Create a new trade based on the executed order.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            bar (Bar): The current price bar.
            size (Optional[ExtendedDecimal]): The size of the trade, if different from the order size.

        Returns:
            Trade: The newly created trade.
        """
        symbol = order.details.ticker
        trade_size = size or order.details.size

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []

        # Always create a new trade
        self.trade_count += 1
        new_trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
        )
        # Set the initial size of the trade
        new_trade.initial_size = trade_size
        new_trade.current_size = trade_size
        self.open_trades[symbol].append(new_trade)

        logger_main.info(
            f"{self.engine._current_timestamp} | Created new trade: \n{new_trade}"
        )
        return new_trade

    def _close_or_reduce_trade(
        self,
        symbol: str,
        size: ExtendedDecimal,
        execution_price: ExtendedDecimal,
        order: Order,
        bar: Bar,
    ) -> Tuple[List[Trade], ExtendedDecimal]:
        """
        Close or reduce existing trades based on the incoming order.

        Args:
            symbol (str): The symbol of the trade.
            size (ExtendedDecimal): The size to close or reduce.
            execution_price (ExtendedDecimal): The price at which to close or reduce the trade.
            order (Order): The order associated with this action.
            bar (Bar): The current price bar.

        Returns:
            Tuple[List[Trade], ExtendedDecimal]: A tuple containing a list of affected trades and the remaining size.
        """
        remaining_size = size
        affected_trades = []

        for trade in self.open_trades[symbol]:
            if remaining_size <= ExtendedDecimal("0"):
                break

            if remaining_size >= trade.current_size:
                # Fully close this trade
                trade.close(order, bar)
            else:
                # Partially close this trade
                trade.close(order, bar, size=remaining_size)

            affected_trades.append(trade)
            remaining_size -= trade.current_size

        # Remove closed trades from open trades
        self.open_trades[symbol] = [
            t for t in self.open_trades[symbol] if t not in affected_trades
        ]
        self.closed_trades.extend(
            [t for t in affected_trades if t.status == Trade.Status.CLOSED]
        )

        return affected_trades, remaining_size

    def _reverse_trade(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Trade:
        """
        Handle the creation of a new trade in the opposite direction after closing existing trades.

        Args:
            order (Order): The order that triggered the trade reversal.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Trade: The newly created trade in the opposite direction.
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        # Close existing trades in the opposite direction
        affected_trades, remaining_size = self._close_or_reduce_trade(
            symbol, size, execution_price, order, bar
        )

        # Update the position
        for trade in affected_trades:
            self._update_position(trade.ticker, -trade.current_size, execution_price)

        # If there's remaining size, create a new trade in the opposite direction
        if remaining_size > ExtendedDecimal("0"):
            new_trade = self._create_or_update_trade(
                order, execution_price, bar, size=remaining_size
            )
            new_trade.reverse(order, bar)  # Use the new reverse method
            self._update_position(
                symbol, remaining_size * direction.value, execution_price
            )
            return new_trade
        else:
            # If no remaining size, return the last affected trade
            return affected_trades[-1] if affected_trades else None

    # endregion

    # region Margin Related Methods

    def _check_margin_requirements(self, order: Order, cost: ExtendedDecimal) -> bool:
        """
        Check if there's sufficient margin to execute the order.

        Args:
            order (Order): The order to check.
            cost (ExtendedDecimal): The cost of the order.

        Returns:
            bool: True if there's sufficient margin, False otherwise.
        """
        if order.details.direction == Order.Direction.LONG:
            required_margin = cost * self.margin_ratio
        else:  # SHORT
            required_margin = (
                cost * self.margin_ratio * ExtendedDecimal("2")
            )  # Higher margin for short selling

        return self.buying_power >= required_margin

    def _update_margin(self, order: Order, cost: ExtendedDecimal) -> None:
        """
        Update margin and buying power after executing an order.

        Args:
            order (Order): The executed order.
            cost (ExtendedDecimal): The cost of the order.
        """
        if order.details.direction == Order.Direction.LONG:
            self.margin_used += cost * self.margin_ratio
        else:  # SHORT
            self.margin_used += cost * self.margin_ratio * ExtendedDecimal("2")

        self.buying_power = (
            self.calculate_equity() - self.margin_used
        ) / self.margin_ratio

    def _check_margin_call(self) -> bool:
        """
        Check if a margin call should be triggered.

        Returns:
            bool: True if a margin call should be triggered, False otherwise.
        """
        equity = self.calculate_equity()
        if self.margin_used:
            if equity / self.margin_used < self.margin_call_threshold:
                logger_main.info("Margin call triggered!")
                return True
        return False

    def _handle_margin_call(self) -> None:
        """
        Handle a margin call by closing positions until margin requirements are met.
        """
        while self._check_margin_call() and self.open_trades:
            largest_trade = max(
                (trade for trades in self.open_trades.values() for trade in trades),
                key=lambda t: abs(t.current_size * t.entry_price),
            )
            self.close_trade(largest_trade, largest_trade.current_price)

    # endregion

    # region Portfolio Analysis and Reporting

    def get_metrics_data(self) -> pd.DataFrame:
        """
        Retrieve the complete metrics DataFrame.

        This method returns the entire metrics DataFrame, including all calculated
        metrics and symbol-specific data.

        Returns:
            pd.DataFrame: The complete metrics DataFrame.
        """
        return self.metrics

    def get_current_equity(self) -> ExtendedDecimal:
        """
        Get the current total equity of the portfolio.

        Returns:
            ExtendedDecimal: The current equity value.
        """
        if len(self.metrics) > 0:
            return ExtendedDecimal(str(self.metrics.iloc[-1]["equity"]))
        return self.initial_capital

    def get_current_cash(self) -> ExtendedDecimal:
        """
        Get the current cash balance of the portfolio.

        Returns:
            ExtendedDecimal: The current cash balance.
        """
        return self.cash

    def get_open_positions(self) -> Dict[str, Dict[str, ExtendedDecimal]]:
        """
        Get the current open positions in the portfolio.

        Returns:
            Dict[str, Dict[str, ExtendedDecimal]]: A dictionary mapping symbols to their position details.
        """
        return self.positions

    def get_total_pnl(self) -> ExtendedDecimal:
        """
        Get the total profit and loss (PnL) of the portfolio.

        This includes both realized and unrealized PnL.

        Returns:
            ExtendedDecimal: The total PnL.
        """
        if len(self.metrics) > 0:
            return ExtendedDecimal(
                str(
                    self.metrics.iloc[-1]["open_pnl"]
                    + self.metrics.iloc[-1]["closed_pnl"]
                )
            )
        return ExtendedDecimal("0")

    # endregion

    # region Utility Methods

    def calculate_equity(self) -> ExtendedDecimal:
        """
        Calculate the total portfolio equity.

        Returns:
            ExtendedDecimal: The total portfolio equity.
        """
        position_value = sum(
            abs(position["size"] * position["current_price"])
            for position in self.positions.values()
            if "current_price" in position
        )
        return self.cash + position_value

    def get_account_value(self) -> ExtendedDecimal:
        """
        Get the current total account value (equity).

        Returns:
            ExtendedDecimal: The current account value.
        """
        return self.calculate_equity()

    def get_position_size(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            ExtendedDecimal: The current position size. Positive for long positions, negative for short positions.
        """
        return self.positions.get(symbol, ExtendedDecimal("0"))

    def get_available_margin(self) -> ExtendedDecimal:
        """
        Get the available margin for new trades.

        Returns:
            ExtendedDecimal: The available margin.
        """
        return self.buying_power

    def get_open_trades(self) -> List[Trade]:
        """
        Get all open trades.

        Returns:
            List[Trade]: A list of all open trades.
        """
        return [trade for trades in self.open_trades.values() for trade in trades]

    def get_closed_trades(self) -> List[Trade]:
        """
        Get all closed trades.

        Returns:
            List[Trade]: A list of all closed trades.
        """
        return self.closed_trades

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.
        """
        return [
            trade for trade in self.closed_trades if trade.strategy_id == strategy_id
        ] + [
            trade
            for trades in self.open_trades.values()
            for trade in trades
            if trade.strategy_id == strategy_id
        ]

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state.
        """
        return {
            "cash": self.cash,
            "equity": self.calculate_equity(),
            "open_positions": self.get_open_positions(),
            "open_trades": {
                symbol: [trade.to_dict() for trade in trades]
                for symbol, trades in self.open_trades.items()
            },
            "pending_orders": [order.to_dict() for order in self.pending_orders],
            "total_trades": self.trade_count,
            "closed_trades": len(self.closed_trades),
            "margin_used": self.margin_used,
            "buying_power": self.buying_power,
        }

    def _generate_order_id(self) -> int:
        """
        Generate a unique order ID.

        Returns:
            int: A unique order ID.
        """
        return hash(f"order_{datetime.now().timestamp()}_{len(self.pending_orders)}")

    # endregion
