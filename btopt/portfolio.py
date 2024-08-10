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
        engine: "EngineType" = None,
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
            engine (Engine): The Engine instance managing the trading system.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.pyramiding = pyramiding
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.engine = engine

        # New attributes for improved equity calculation
        self.asset_value = ExtendedDecimal("0")
        self.liabilities = ExtendedDecimal("0")
        self.buying_power = initial_capital

        # Existing attributes
        self.positions: Dict[str, ExtendedDecimal] = {}
        self.avg_entry_prices: Dict[str, ExtendedDecimal] = {}
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0
        self.margin_used = ExtendedDecimal("0")
        self.limit_exit_orders: List[Order] = []

        self.updated_orders: List[Order] = []
        self.updated_trades: List[Trade] = []

        # Initialize the metrics DataFrame with new columns
        self.metrics = pd.DataFrame(
            columns=[
                "timestamp",
                "cash",
                "equity",
                "asset_value",
                "liabilities",
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
        self,
        timestamp: pd.Timestamp,
        market_data: Dict[str, Dict[Timeframe, Bar]],
    ) -> None:
        """
        Update the portfolio metrics based on the current market data.

        This method calculates and stores essential metrics including cash, equity,
        asset value, liabilities, open/closed PnL, and portfolio returns.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.

        Returns:
            None
        """
        open_pnl = ExtendedDecimal("0")
        self.asset_value = ExtendedDecimal("0")
        self.liabilities = ExtendedDecimal("0")

        for symbol, trades in self.open_trades.items():
            timeframe = min(market_data[symbol].keys())
            current_bar = market_data[symbol][timeframe]
            for trade in trades:
                trade.update(current_bar)
                open_pnl += trade.metrics.pnl
                if trade.direction == Order.Direction.LONG:
                    self.asset_value += trade.current_size * current_bar.close
                else:  # SHORT
                    self.liabilities += trade.current_size * current_bar.close

        closed_pnl = sum(trade.metrics.pnl for trade in self.closed_trades)
        equity = self.calculate_equity()

        # Calculate portfolio return
        if not self.metrics.empty:
            previous_equity = self.metrics.iloc[-1]["equity"]
            portfolio_return = (equity - previous_equity) / previous_equity
        else:
            portfolio_return = ExtendedDecimal("0")

        # Create a new row for the metrics DataFrame
        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "cash": [self.cash],
                "equity": [equity],
                "asset_value": [self.asset_value],
                "liabilities": [self.liabilities],
                "open_pnl": [open_pnl],
                "closed_pnl": [closed_pnl],
                "portfolio_return": [portfolio_return],
            }
        )

        # Append the new row to the metrics DataFrame
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

        # Update buying power
        self._update_buying_power()

    def _get_current_price(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current price for a symbol.

        Args:
            symbol (str): The symbol to get the price for.

        Returns:
            ExtendedDecimal: The current price of the symbol.
        """
        if self.engine and hasattr(self.engine, "_current_market_data"):
            market_data = self.engine._current_market_data[symbol]
            current_data = market_data[min(market_data.keys())]
            if current_data is not None:
                return ExtendedDecimal(str(current_data["close"]))

        # If we can't get the current price, use the last known price
        logger_main.log_and_raise(
            ValueError(f"Unable to get current price for {symbol} {self.engine}")
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

        This method handles the execution of an order, updating cash, asset value,
        liabilities, and creating or updating trades as necessary. It also adds
        the affected trade to the updated_trades list for strategy notification.

        Args:
            order (Order): The order to execute.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing:
                - bool: True if the order was executed successfully, False otherwise.
                - Optional[Trade]: The resulting Trade object if applicable, None otherwise.

        Raises:
            ValueError: If there's insufficient margin to execute the order.
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        cost = execution_price * size
        commission = cost * self.commission_rate

        if not self._check_margin_requirements(order, cost):
            logger_main.info(f"Insufficient margin to execute order: {order}")
            return False, None

        # Update cash, asset value, and liabilities based on order direction
        if direction == Order.Direction.LONG:
            self.cash -= cost + commission
            self.asset_value += cost
        else:  # SHORT
            self.cash += cost - commission
            self.liabilities += cost

        self._update_margin(order, cost)

        # Check if this order would result in a trade reversal
        current_position = self.positions.get(symbol, ExtendedDecimal("0"))
        is_reversal = (current_position > 0 and direction == Order.Direction.SHORT) or (
            current_position < 0 and direction == Order.Direction.LONG
        )

        if is_reversal:
            trade = self._reverse_trade(order, execution_price, bar)
        else:
            trade = self._create_or_update_trade(order, execution_price, bar)

        # Add the affected trade to updated_trades for strategy notification
        if trade:
            self._add_to_updated_trades(trade)

        # Update order status based on fill
        if order.get_remaining_size() > ExtendedDecimal("0"):
            order.partial_fill(execution_price, size, bar.timestamp)
        else:
            order.fill(execution_price, bar.timestamp, size)

        self.updated_orders.append(order)

        # Update buying power
        self._update_buying_power()

        logger_main.info(f"Executed order: {order}, resulting trade: {trade}")
        return True, trade

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

    def close_trade(self, trade: Trade, current_price: ExtendedDecimal) -> None:
        """
        Close a specific trade and update the portfolio accordingly.

        This method handles the closing of a trade, updating cash, asset value,
        liabilities, and moving the trade from open to closed trades. It also adds
        the closed trade to the updated_trades list for strategy notification.

        Args:
            trade (Trade): The trade to close.
            current_price (ExtendedDecimal): The current market price to close the trade at.
        """
        trade.close(None, current_price)
        if trade.ticker in self.open_trades:
            self.open_trades[trade.ticker].remove(trade)
            if not self.open_trades[trade.ticker]:
                del self.open_trades[trade.ticker]
        self.closed_trades.append(trade)

        # Update cash, asset value, and liabilities
        trade_value = trade.current_size * current_price
        commission = trade_value * self.commission_rate

        if trade.direction == Order.Direction.LONG:
            self.cash += trade_value - commission
            self.asset_value -= trade_value
        else:  # SHORT
            self.cash -= trade_value + commission
            self.liabilities -= trade_value

        self._update_position(trade.ticker, -trade.current_size, current_price)
        self._update_buying_power()

        # Add the closed trade to updated_trades for strategy notification
        self._add_to_updated_trades(trade)

        logger_main.info(f"Closed trade: {trade}")

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
        self, symbol: str, position_change: ExtendedDecimal, price: ExtendedDecimal
    ) -> None:
        """
        Update position and average entry price for a symbol.

        Args:
            symbol (str): The symbol to update.
            position_change (ExtendedDecimal): The change in position size.
            price (ExtendedDecimal): The price at which the position change occurred.
        """
        current_position = self.positions.get(symbol, ExtendedDecimal("0"))
        new_position = current_position + position_change

        if current_position == ExtendedDecimal("0"):
            self.avg_entry_prices[symbol] = price
        else:
            current_value = current_position * self.avg_entry_prices[symbol]
            new_value = abs(position_change) * price
            self.avg_entry_prices[symbol] = (current_value + new_value) / abs(
                new_position
            )

        self.positions[symbol] = new_position

        if new_position == ExtendedDecimal("0"):
            del self.positions[symbol]
            del self.avg_entry_prices[symbol]

    def _update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Update all open trades based on current market data.

        This method iterates through all open trades, updates them with the latest
        market data, and adds any modified trades to the updated_trades list for
        strategy notification.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.
        """
        for symbol, trades in self.open_trades.items():
            timeframe = min(market_data[symbol].keys())
            current_bar = market_data[symbol][timeframe]
            for trade in trades:
                # Store the trade's state before update
                pre_update_state = trade.to_dict()

                trade.update(current_bar)

                # Check if the trade state has changed
                if trade.to_dict() != pre_update_state:
                    self._add_to_updated_trades(trade)

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

        logger_main.info(f"Created new trade: {new_trade}")
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

        This method processes the closure or reduction of trades and adds
        affected trades to the updated_trades list for strategy notification.

        Args:
            symbol (str): The symbol of the trade.
            size (ExtendedDecimal): The size to close or reduce.
            execution_price (ExtendedDecimal): The price at which to close or reduce the trade.
            order (Order): The order associated with this action.
            bar (Bar): The current price bar.

        Returns:
            Tuple[List[Trade], ExtendedDecimal]: A tuple containing:
                - List[Trade]: A list of affected trades.
                - ExtendedDecimal: The remaining size after closing or reducing trades.
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

            # Add the affected trade to updated_trades for strategy notification
            self._add_to_updated_trades(trade)

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
    ) -> Optional[Trade]:
        """
        Handle the creation of a new trade in the opposite direction after closing existing trades.

        This method processes trade reversal, closes existing trades, creates a new trade
        if necessary, and adds affected trades to the updated_trades list for strategy notification.

        Args:
            order (Order): The order that triggered the trade reversal.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Optional[Trade]: The newly created trade in the opposite direction or the last affected trade.
                             Returns None if no trades were affected or created.
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        # Close existing trades in the opposite direction
        affected_trades, remaining_size = self._close_or_reduce_trade(
            symbol, size, execution_price, order, bar
        )

        # Update the position for closed trades
        for trade in affected_trades:
            self._update_position(trade.ticker, -trade.current_size, execution_price)
            self._add_to_updated_trades(trade)

        # If there's remaining size, create a new trade in the opposite direction
        if remaining_size > ExtendedDecimal("0"):
            new_trade = self._create_or_update_trade(
                order, execution_price, bar, size=remaining_size
            )
            new_trade.reverse(order, bar)
            self._update_position(
                symbol, remaining_size * direction.value, execution_price
            )
            # Add the new trade to updated_trades for strategy notification
            self._add_to_updated_trades(new_trade)
            return new_trade
        elif affected_trades:
            # If no remaining size, return the last affected trade
            return affected_trades[-1]
        else:
            # If no trades were affected or created
            return None

    def _add_to_updated_trades(self, trade: Trade) -> None:
        """
        Add a trade to the updated_trades list if it's not already present.

        This method ensures that each trade is only added once to the updated_trades
        list during a single update cycle.

        Args:
            trade (Trade): The trade to be added to the updated_trades list.
        """
        if trade not in self.updated_trades:
            self.updated_trades.append(trade)
            logger_main.debug(f"Added trade {trade.id} to updated_trades list")

    def clear_updated_orders_and_trades(self) -> None:
        """
        Clear the lists of updated orders and trades.

        This method should be called by the Engine after notifying strategies
        of order and trade updates. It resets the updated_orders and updated_trades
        lists, preparing them for the next timestamp cycle.
        """
        self.updated_orders.clear()
        self.updated_trades.clear()
        logger_main.debug("Cleared updated orders and trades lists")

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

        This method calculates the required margin based on the order type and updates
        the margin_used attribute. It also recalculates the buying power based on
        the new equity and margin situation.

        Args:
            order (Order): The executed order.
            cost (ExtendedDecimal): The cost of the order.
        """
        if order.details.direction == Order.Direction.LONG:
            self.margin_used += cost * self.margin_ratio
        else:  # SHORT
            self.margin_used += cost * self.margin_ratio * ExtendedDecimal("2")

        self._update_buying_power()

    def _update_buying_power(self) -> None:
        """
        Update the buying power based on current equity and margin used.

        This method recalculates the available buying power considering the current
        equity, margin used, and margin ratio.
        """
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

        This method closes the largest trades until the margin call is satisfied,
        and adds all closed trades to the updated_trades list for strategy notification.
        """
        while self._check_margin_call() and self.open_trades:
            largest_trade = max(
                (trade for trades in self.open_trades.values() for trade in trades),
                key=lambda t: abs(t.current_size * t.entry_price),
            )
            self.close_trade(largest_trade, largest_trade.current_price)
            # The close_trade method now handles adding to updated_trades

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

    def get_open_positions(self) -> Dict[str, ExtendedDecimal]:
        """
        Get the current open positions in the portfolio.

        Returns:
            Dict[str, ExtendedDecimal]: A dictionary mapping symbols to their position sizes.
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

        This method computes the total equity by summing the cash balance,
        the value of long positions (asset_value), and subtracting the value
        of short positions (liabilities).

        Returns:
            ExtendedDecimal: The total portfolio equity.
        """
        return self.cash + self.asset_value - self.liabilities

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

        This method returns a comprehensive dictionary containing the current
        state of the portfolio, including cash, equity, asset value, liabilities,
        open trades, pending orders, and other relevant metrics.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state.
        """
        return {
            "cash": self.cash,
            "equity": self.calculate_equity(),
            "asset_value": self.asset_value,
            "liabilities": self.liabilities,
            "open_trades": {
                symbol: [trade.to_dict() for trade in trades]
                for symbol, trades in self.open_trades.items()
            },
            "pending_orders": [order.to_dict() for order in self.pending_orders],
            "total_trades": self.trade_count,
            "closed_trades": len(self.closed_trades),
            "margin_used": self.margin_used,
            "buying_power": self.buying_power,
            "margin_ratio": self.margin_ratio,
        }

    def _generate_order_id(self) -> int:
        """
        Generate a unique order ID.

        Returns:
            int: A unique order ID.
        """
        return hash(f"order_{datetime.now().timestamp()}_{len(self.pending_orders)}")

    # endregion
