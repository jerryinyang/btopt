from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails
from .trade import Trade


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

    Attributes:
        initial_capital (Decimal): The starting capital of the portfolio.
        cash (Decimal): The current cash balance.
        commission_rate (Decimal): The commission rate for trades.
        slippage (Decimal): The slippage rate for trades.
        pyramiding (int): The maximum number of allowed positions per symbol.
        margin_ratio (Decimal): The required margin ratio for trades.
        margin_call_threshold (Decimal): The threshold for triggering a margin call.
        positions (Dict[str, Decimal]): Current positions for each symbol.
        avg_entry_prices (Dict[str, Decimal]): Average entry prices for each symbol.
        open_trades (Dict[str, List[Trade]]): Open trades grouped by symbol.
        closed_trades (List[Trade]): List of closed trades.
        pending_orders (List[Order]): List of pending orders.
        limit_exit_orders (List[Order]): List of pending limit exit orders.
        trade_count (int): Total number of trades executed.
        margin_used (Decimal): Amount of margin currently in use.
        buying_power (Decimal): Available buying power for new trades.
        metrics (pd.DataFrame): DataFrame storing portfolio metrics over time.
        peak_equity (Decimal): Highest equity value reached.
        max_drawdown (Decimal): Maximum drawdown experienced.

    Usage:
        portfolio = Portfolio(initial_capital=Decimal("100000"), commission_rate=Decimal("0.001"))
        portfolio.create_order("AAPL", Order.Direction.LONG, 100, Order.ExecType.MARKET)
        portfolio.update(timestamp, market_data)
        performance = portfolio.get_performance_metrics()
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0"),
        pyramiding: int = 1,
        margin_ratio: Decimal = Decimal("0.5"),
        margin_call_threshold: Decimal = Decimal("0.3"),
    ):
        """
        Initialize the Portfolio with given parameters.

        Args:
            initial_capital (Decimal): The starting capital of the portfolio.
            commission_rate (Decimal): The commission rate for trades.
            slippage (Decimal): The slippage rate for trades.
            pyramiding (int): The maximum number of allowed positions per symbol.
            margin_ratio (Decimal): The required margin ratio for trades.
            margin_call_threshold (Decimal): The threshold for triggering a margin call.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.pyramiding = pyramiding
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.positions: Dict[str, Decimal] = {}  # Current positions for each symbol
        self.avg_entry_prices: Dict[
            str, Decimal
        ] = {}  # Average entry prices for each symbol
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0
        self.margin_used = Decimal("0")
        self.buying_power = initial_capital / margin_ratio
        self.limit_exit_orders: List[Order] = []

        # DataFrame to store portfolio metrics over time
        self.metrics = pd.DataFrame(
            columns=[
                "timestamp",
                "cash",
                "equity",
                "open_pnl",
                "closed_pnl",
                "commission",
                "slippage",
                "drawdown",
                "max_drawdown",
            ]
        )
        self.peak_equity = initial_capital
        self.max_drawdown = Decimal("0")

        self.updated_orders: List[Order] = []
        self.updated_trades: List[Trade] = []

    # region Portfolio Update and Market Data Processing

    def update(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> None:
        """
        Update the portfolio state based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.
        """
        self._process_pending_orders(timestamp, market_data)
        self._update_open_trades(timestamp, market_data)
        self._update_metrics(timestamp, market_data)

    def _update_metrics(
        self,
        timestamp: pd.Timestamp,
        market_data: Dict[str, Dict[Timeframe, np.ndarray]],
    ) -> None:
        """
        Update portfolio metrics based on current market data.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.
        """
        open_pnl = Decimal("0")
        commission = Decimal("0")
        slippage = Decimal("0")

        # Calculate open P&L, commission, and slippage for all open trades
        for symbol, trades in self.open_trades.items():
            current_price = market_data[symbol][Timeframe("1m")][3]  # Close price
            for trade in trades:
                trade.update(current_price)
                open_pnl += trade.metrics.pnl
                commission += trade.metrics.commission
                slippage += trade.metrics.slippage

        closed_pnl = sum(trade.metrics.pnl for trade in self.closed_trades)
        equity = self.cash + open_pnl + closed_pnl
        drawdown = self.peak_equity - equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self.peak_equity = max(self.peak_equity, equity)

        # Append new row to metrics DataFrame
        self.metrics = self.metrics.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "equity": equity,
                "open_pnl": open_pnl,
                "closed_pnl": closed_pnl,
                "commission": commission,
                "slippage": slippage,
                "drawdown": drawdown,
                "max_drawdown": self.max_drawdown,
            },
            ignore_index=True,
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
            size=Decimal(str(size)),
            price=Decimal(str(price)) if price is not None else None,
            exectype=order_type,
            timestamp=datetime.now(),
            timeframe=kwargs.get("timeframe"),
            **kwargs,
        )
        order = Order(order_id=self._generate_order_id(), details=order_details)
        self.add_pending_order(order)
        return order

    def execute_order(
        self, order: Order, execution_price: Decimal
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update the portfolio accordingly.

        Args:
            order (Order): The order to execute.
            execution_price (Decimal): The price at which the order is executed.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing a boolean indicating if the order was executed
                                          successfully, and the resulting Trade object if applicable.
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        cost = execution_price * size
        commission = cost * self.commission_rate

        # Check if there's enough margin to execute the order
        if not self._check_margin_requirements(order, cost):
            logger_main.log_and_print(f"Insufficient margin to execute order: {order}")
            return False, None

        self._update_margin(order, cost)

        # Update cash based on order direction
        if direction == Order.Direction.LONG:
            self.cash -= cost + commission
        else:  # SHORT
            self.cash += cost - commission

        # Update position
        position_change = size if direction == Order.Direction.LONG else -size
        self._update_position(symbol, position_change, execution_price)

        # Handle different order types
        if order.family_role == Order.FamilyRole.PARENT:
            trade = self._create_or_update_trade(order, execution_price)
            # Add child orders to the appropriate list
            for child_order in order.children:
                self.add_pending_order(child_order)
        elif order.family_role == Order.FamilyRole.CHILD_EXIT:
            trade = self._close_or_reduce_trade(symbol, size, execution_price, order)
        else:
            trade = self._create_or_update_trade(order, execution_price)

        # Update order status
        order.fill(execution_price, datetime.now(), size)

        self.updated_orders.append(order)
        if trade:
            self.updated_trades.append(trade)

        logger_main.log_and_print(
            f"Executed order: {order}, resulting trade: {trade}", level="info"
        )
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
        logger_main.log_and_print(f"Added pending order: {order}", level="info")

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
            logger_main.log_and_print(
                f"Failed to cancel order (not found in pending orders): {order}",
                level="warning",
            )
            return False

        order.status = Order.Status.CANCELED
        self.updated_orders.append(order)
        logger_main.log_and_print(f"Cancelled order: {order}", level="info")
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
                logger_main.log_and_print(f"Modified order: {order}", level="info")
                return True
        logger_main.log_and_print(
            f"Order with ID {order_id} not found in pending orders."
        )
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
            logger_main.log_and_print(
                f"Closed positions for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
                level="info",
            )
        else:
            logger_main.log_and_print(
                f"No positions to close for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
                level="info",
            )

        return closed_any

    def get_closed_trades(
        self, strategy_id: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Trade]:
        """
        Get closed trades filtered by strategy ID and/or symbol.

        Args:
            strategy_id (Optional[str]): The strategy ID to filter by.
            symbol (Optional[str]): The symbol to filter by.

        Returns:
            List[Trade]: A list of closed trades matching the filter criteria.
        """
        return [
            trade
            for trade in self.closed_trades
            if (strategy_id is None or trade.strategy_id == strategy_id)
            and (symbol is None or trade.ticker == symbol)
        ]

    def get_pending_orders(
        self, strategy_id: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get pending orders filtered by strategy ID and/or symbol.

        Args:
            strategy_id (Optional[str]): The strategy ID to filter by.
            symbol (Optional[str]): The symbol to filter by.

        Returns:
            List[Order]: A list of pending orders matching the filter criteria.
        """
        return [
            order
            for order in self.pending_orders
            if (strategy_id is None or order.details.strategy_id == strategy_id)
            and (symbol is None or order.details.ticker == symbol)
        ]

    def get_updated_orders(self) -> List[Order]:
        """
        Get and clear the list of updated orders.

        Returns:
            List[Order]: A list of orders that have been updated since the last call.
        """
        updated = self.updated_orders.copy()
        self.updated_orders.clear()
        return updated

    def get_updated_trades(self) -> List[Trade]:
        """
        Get and clear the list of updated trades.

        Returns:
            List[Trade]: A list of trades that have been updated since the last call.
        """
        updated = self.updated_trades.copy()
        self.updated_trades.clear()
        return updated

    def _update_position(self, symbol: str, position_change: Decimal, price: Decimal):
        """
        Update position and average entry price for a symbol.

        Args:
            symbol (str): The symbol to update.
            position_change (Decimal): The change in position size.
            price (Decimal): The price at which the position change occurred.
        """
        current_position = self.positions.get(symbol, Decimal("0"))
        new_position = current_position + position_change

        if current_position == Decimal("0"):
            self.avg_entry_prices[symbol] = price
        else:
            # Calculate new average entry price
            current_value = current_position * self.avg_entry_prices[symbol]
            new_value = abs(position_change) * price
            self.avg_entry_prices[symbol] = (current_value + new_value) / abs(
                new_position
            )

        self.positions[symbol] = new_position

        if new_position == Decimal("0"):
            del self.positions[symbol]
            del self.avg_entry_prices[symbol]

    def _update_open_trades(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> None:
        """
        Update all open trades based on current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.
        """
        for symbol, trades in self.open_trades.items():
            current_price = market_data[symbol][Timeframe("1m")][3]  # Close price
            for trade in trades:
                trade.update(current_price)
                self.updated_trades.append(trade)

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> List[Tuple[Order, bool, Optional[Trade]]]:
        """
        Process all pending orders based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.

        Returns:
            List[Tuple[Order, bool, Optional[Trade]]]: A list of tuples containing the processed order,
            a boolean indicating if it was executed, and the resulting trade (if any).
        """
        results = []

        # Process regular pending orders
        results.extend(
            self._process_order_list(self.pending_orders, timestamp, market_data)
        )

        # Process limit exit orders
        results.extend(
            self._process_order_list(self.limit_exit_orders, timestamp, market_data)
        )

        return results

    def _process_order_list(
        self,
        order_list: List[Order],
        timestamp: datetime,
        market_data: Dict[str, Dict[Timeframe, np.ndarray]],
    ) -> List[Tuple[Order, bool, Optional[Trade]]]:
        """
        Process a list of orders based on the current market data.

        Args:
            order_list (List[Order]): The list of orders to process.
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.

        Returns:
            List[Tuple[Order, bool, Optional[Trade]]]: A list of tuples containing the processed order,
            a boolean indicating if it was executed, and the resulting trade (if any).
        """
        results = []
        for order in order_list[:]:  # Create a copy to iterate over
            symbol = order.details.ticker
            current_price = market_data[symbol][order.details.timeframe][
                3
            ]  # Close price
            is_filled, fill_price = order.is_filled(current_price)
            if is_filled:
                executed, trade = self.execute_order(order, fill_price)
                results.append((order, executed, trade))
                order_list.remove(order)
            elif order.is_expired(timestamp):
                order_list.remove(order)
                results.append((order, False, None))

        return results

    def _check_order_fill(
        self, order: Order, current_price: Decimal
    ) -> Tuple[bool, Optional[Decimal]]:
        """
        Check if an order should be filled based on current price and order type.

        Args:
            order (Order): The order to check.
            current_price (Decimal): The current market price.

        Returns:
            Tuple[bool, Optional[Decimal]]: A tuple containing a boolean indicating if the order should be filled,
            and the fill price (if applicable).
        """
        if order.details.exectype == Order.ExecType.MARKET:
            return True, self._apply_slippage(current_price, order.details.size)

        if order.details.exectype == Order.ExecType.LIMIT:
            if (
                order.details.direction == Order.Direction.LONG
                and current_price <= order.details.price
            ) or (
                order.details.direction == Order.Direction.SHORT
                and current_price >= order.details.price
            ):
                return True, order.details.price

        if order.details.exectype == Order.ExecType.STOP:
            if (
                order.details.direction == Order.Direction.LONG
                and current_price >= order.details.price
            ) or (
                order.details.direction == Order.Direction.SHORT
                and current_price <= order.details.price
            ):
                return True, current_price

        return False, None

    def _apply_slippage(self, price: Decimal, size: Decimal) -> Decimal:
        """
        Apply slippage to the given price based on order size and volatility.

        Args:
            price (Decimal): The original price.
            size (Decimal): The order size.

        Returns:
            Decimal: The price after applying slippage.
        """
        # This is a simplified slippage model. You might want to implement a more sophisticated one.
        slippage_factor = self.slippage * (
            1 + (size / Decimal("10000"))
        )  # Adjust based on your typical order sizes
        return price * (1 + slippage_factor)

    def _create_or_update_trade(self, order: Order, execution_price: Decimal) -> Trade:
        """
        Create a new trade or update existing one based on the executed order.

        Args:
            order (Order): The executed order.
            execution_price (Decimal): The price at which the order was executed.

        Returns:
            Trade: The created or updated trade.
        """
        symbol = order.details.ticker
        direction = order.details.direction
        size = order.details.size

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []

        # Check if this order closes or reduces an existing position
        if (
            direction == Order.Direction.LONG
            and self.positions.get(symbol, Decimal("0")) < Decimal("0")
        ) or (
            direction == Order.Direction.SHORT
            and self.positions.get(symbol, Decimal("0")) > Decimal("0")
        ):
            # This is a closing or reducing order
            return self._close_or_reduce_trade(symbol, size, execution_price, order)
        else:
            # This is a new or increasing position order
            return self._open_or_increase_trade(symbol, size, execution_price, order)

    def _close_or_reduce_trade(
        self, symbol: str, size: Decimal, price: Decimal, order: Order
    ) -> Trade:
        """
        Close or reduce an existing trade.

        Args:
            symbol (str): The symbol of the trade.
            size (Decimal): The size to close or reduce.
            price (Decimal): The price at which to close or reduce the trade.
            order (Order): The order associated with this action.

        Returns:
            Trade: The last affected trade.
        """
        remaining_size = size
        closed_trades = []

        for trade in self.open_trades[symbol]:
            if remaining_size <= Decimal("0"):
                break

            if remaining_size >= trade.current_size:
                # Fully close this trade
                trade.close(order, price)
                closed_trades.append(trade)
                remaining_size -= trade.current_size
            else:
                # Partially close this trade
                trade.close(order, price, remaining_size)
                remaining_size = Decimal("0")

        # Remove closed trades from open trades
        self.open_trades[symbol] = [
            t for t in self.open_trades[symbol] if t not in closed_trades
        ]
        self.closed_trades.extend(closed_trades)

        if remaining_size > Decimal("0"):
            # If there's remaining size, it means we've reversed the position
            return self._open_or_increase_trade(symbol, remaining_size, price, order)
        else:
            return closed_trades[-1] if closed_trades else None

    def _open_or_increase_trade(
        self, symbol: str, size: Decimal, price: Decimal, order: Order
    ) -> Trade:
        """
        Open a new trade or increase an existing one.

        Args:
            symbol (str): The symbol of the trade.
            size (Decimal): The size of the trade.
            price (Decimal): The price at which to open or increase the trade.
            order (Order): The order associated with this action.

        Returns:
            Trade: The newly created or updated trade.
        """
        self.trade_count += 1
        trade = Trade(
            self.trade_count,
            order,
            price,
            commission_rate=self.commission_rate,
        )
        self.open_trades[symbol].append(trade)
        return trade

    # endregion

    # region Margin Related Methods

    def _check_margin_requirements(self, order: Order, cost: Decimal) -> bool:
        """
        Check if there's sufficient margin to execute the order.

        Args:
            order (Order): The order to check.
            cost (Decimal): The cost of the order.

        Returns:
            bool: True if there's sufficient margin, False otherwise.
        """
        if order.details.direction == Order.Direction.LONG:
            required_margin = cost * self.margin_ratio
        else:  # SHORT
            required_margin = (
                cost * self.margin_ratio * Decimal("2")
            )  # Higher margin for short selling

        return self.buying_power >= required_margin

    def _update_margin(self, order: Order, cost: Decimal) -> None:
        """
        Update margin and buying power after executing an order.

        Args:
            order (Order): The executed order.
            cost (Decimal): The cost of the order.
        """
        if order.details.direction == Order.Direction.LONG:
            self.margin_used += cost * self.margin_ratio
        else:  # SHORT
            self.margin_used += cost * self.margin_ratio * Decimal("2")

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
        if equity / self.margin_used < self.margin_call_threshold:
            logger_main.log_and_print("Margin call triggered!")
            return True
        return False

    def _handle_margin_call(self) -> None:
        """
        Handle a margin call by closing positions until margin requirements are met.
        """
        while self._check_margin_call() and self.open_trades:
            # Close the largest open trade
            largest_trade = max(
                (trade for trades in self.open_trades.values() for trade in trades),
                key=lambda t: abs(t.current_size * t.entry_price),
            )
            self.close_trade(largest_trade, largest_trade.current_price)

    # endregion

    # region Portfolio Analysis and Reporting

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return various performance metrics for the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics.
        """
        total_return = self.calculate_total_return()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.get_max_drawdown()
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.closed_trades),
        }

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete trade history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a trade.
        """
        return [trade.to_dict() for trade in self.closed_trades]

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the equity curve data.
        """
        return self.metrics[["timestamp", "equity"]]

    def calculate_total_return(self) -> Decimal:
        """
        Calculate the total return of the portfolio.

        Returns:
            Decimal: The total return as a percentage.
        """
        final_equity = self.calculate_equity()
        return (final_equity - self.initial_capital) / self.initial_capital * 100

    def calculate_sharpe_ratio(
        self, risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Calculate the Sharpe ratio of the portfolio.

        Args:
            risk_free_rate (Decimal): The risk-free rate, default is 2%.

        Returns:
            Decimal: The calculated Sharpe ratio.
        """
        if len(self.metrics) < 2:
            return Decimal("0")

        returns = self.metrics["equity"].pct_change().dropna()
        excess_returns = returns - (
            risk_free_rate / 252
        )  # Assuming daily returns and 252 trading days
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        return Decimal(str(sharpe_ratio))

    def get_max_drawdown(self) -> Decimal:
        """
        Get the maximum drawdown experienced by the portfolio.

        Returns:
            Decimal: The maximum drawdown as a percentage.
        """
        return self.max_drawdown / self.peak_equity * 100

    def calculate_win_rate(self) -> Decimal:
        """
        Calculate the win rate of closed trades.

        Returns:
            Decimal: The win rate as a percentage.
        """
        if not self.closed_trades:
            return Decimal("0")

        winning_trades = sum(1 for trade in self.closed_trades if trade.metrics.pnl > 0)
        return Decimal(winning_trades) / Decimal(len(self.closed_trades)) * 100

    def calculate_profit_factor(self) -> Decimal:
        """
        Calculate the profit factor of the portfolio.

        Returns:
            Decimal: The profit factor.
        """
        total_profit = sum(
            trade.metrics.pnl for trade in self.closed_trades if trade.metrics.pnl > 0
        )
        total_loss = abs(
            sum(
                trade.metrics.pnl
                for trade in self.closed_trades
                if trade.metrics.pnl < 0
            )
        )

        if total_loss == 0:
            return Decimal("inf") if total_profit > 0 else Decimal("0")

        return total_profit / total_loss

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state.
        """
        return {
            "cash": self.cash,
            "equity": self.calculate_equity(),
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

    def get_account_value(self) -> Decimal:
        """
        Get the current total account value (equity).

        Returns:
            Decimal: The current account value.
        """
        return self.calculate_equity()

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            Decimal: The current position size. Positive for long positions, negative for short positions.
        """
        if symbol in self.open_trades:
            return sum(trade.current_size for trade in self.open_trades[symbol])
        return Decimal("0")

    def get_available_margin(self) -> Decimal:
        """
        Get the available margin for new trades.

        Returns:
            Decimal: The available margin.
        """
        return self.buying_power

    # endregion

    # region Utility Methods

    def calculate_equity(self) -> Decimal:
        """
        Calculate the total portfolio equity.

        Returns:
            Decimal: The total portfolio equity.
        """
        open_pnl = sum(
            trade.metrics.pnl
            for trades in self.open_trades.values()
            for trade in trades
        )
        closed_pnl = sum(trade.metrics.pnl for trade in self.closed_trades)
        return self.cash + open_pnl + closed_pnl

    def close_all_trades(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> None:
        """
        Close all open trades.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.
        """
        for symbol, trades in list(self.open_trades.items()):
            current_price = market_data[symbol][Timeframe("1m")][3]  # Close price
            for trade in trades[:]:
                self.close_trade(trade, current_price)
        logger_main.log_and_print("Closed all open trades.", level="info")

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the portfolio.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.initial_capital = Decimal(
            str(config.get("initial_capital", self.initial_capital))
        )
        self.commission_rate = Decimal(
            str(config.get("commission_rate", self.commission_rate))
        )
        self.slippage = Decimal(str(config.get("slippage", self.slippage)))
        self.pyramiding = config.get("pyramiding", self.pyramiding)
        self.margin_ratio = Decimal(str(config.get("margin_ratio", self.margin_ratio)))
        self.margin_call_threshold = Decimal(
            str(config.get("margin_call_threshold", self.margin_call_threshold))
        )
        logger_main.log_and_print("Portfolio configuration set.", level="info")

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
        )
        logger_main.log_and_print("Portfolio reset to initial state.", level="info")

    def _generate_order_id(self) -> int:
        """
        Generate a unique order ID.

        Returns:
            int: A unique order ID.
        """
        return hash(f"order_{datetime.now().timestamp()}_{len(self.pending_orders)}")

    # endregion
