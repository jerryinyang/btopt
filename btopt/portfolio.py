from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails
from .reporter import Reporter
from .trade import Trade
from .types import EngineType


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
        reporter (Reporter): Instance of the Reporter class for generating reports and visualizations.
        engine (Engine): Instance of the Engine class managing the trading system.

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
        engine: EngineType = None,
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
        self.positions: Dict[str, Decimal] = {}
        self.avg_entry_prices: Dict[str, Decimal] = {}
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0
        self.margin_used = Decimal("0")
        self.buying_power = initial_capital / margin_ratio
        self.limit_exit_orders: List[Order] = []

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

        self.reporter: Optional[Reporter] = None

    # region Initialization and Configuration

    def initialize_reporter(self) -> None:
        """
        Initialize the Reporter instance for this Portfolio.
        """
        self.reporter = Reporter(self, self.engine)
        logger_main.log_and_print("Reporter initialized for Portfolio.", level="info")

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
            self.engine,
        )
        logger_main.log_and_print("Portfolio reset to initial state.", level="info")

    # endregion

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

        # Clear the Reporter's cache after updating metrics
        if self.reporter:
            self.reporter.clear_cache()

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

        new_row = pd.DataFrame(
            [
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
                }
            ]
        )

        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

    def _get_current_price(self, symbol: str) -> Decimal:
        """
        Get the current price for a symbol.

        Args:
            symbol (str): The symbol to get the price for.

        Returns:
            Decimal: The current price of the symbol.
        """
        if self.engine and hasattr(self.engine, "_dataview"):
            # Assuming the engine has a _dataview attribute with market data
            current_data = self.engine._dataview.get_last_data(symbol)
            if current_data is not None:
                return Decimal(str(current_data["close"]))

        # If we can't get the current price, use the last known price
        for trade in self.open_trades.get(symbol, []):
            return trade.current_price

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
            size=Decimal(str(size)),
            price=Decimal(str(price)) if price is not None else None,
            exectype=order_type,
            timestamp=datetime.now(),
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

        if not self._check_margin_requirements(order, cost):
            logger_main.log_and_print(
                f"Insufficient margin to execute order: {order}", level="warning"
            )
            return False, None

        self._update_margin(order, cost)

        if direction == Order.Direction.LONG:
            self.cash -= cost + commission
        else:  # SHORT
            self.cash += cost - commission

        position_change = size if direction == Order.Direction.LONG else -size
        self._update_position(symbol, position_change, execution_price)

        if order.family_role == Order.FamilyRole.PARENT:
            trade = self._create_or_update_trade(order, execution_price)
            for child_order in order.children:
                self.add_pending_order(child_order)
        elif order.family_role == Order.FamilyRole.CHILD_EXIT:
            trade = self._close_or_reduce_trade(symbol, size, execution_price, order)
        else:
            trade = self._create_or_update_trade(order, execution_price)

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
            f"Order with ID {order_id} not found in pending orders.", level="warning"
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

        logger_main.log_and_print("Closed all open positions.", level="info")

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

    def close_trade(self, trade: Trade, current_price: Decimal) -> None:
        """
        Close a specific trade.

        Args:
            trade (Trade): The trade to close.
            current_price (Decimal): The current market price to close the trade at.
        """
        trade.close(None, current_price)
        if trade.ticker in self.open_trades:
            self.open_trades[trade.ticker].remove(trade)
            if not self.open_trades[trade.ticker]:
                del self.open_trades[trade.ticker]
        self.closed_trades.append(trade)
        self._update_position(trade.ticker, -trade.current_size, current_price)
        logger_main.log_and_print(f"Closed trade: {trade}", level="info")

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> None:
        """
        Process all pending orders based on the current market data.

        This method handles cases where an order's timeframe might be None by using
        the lowest available timeframe for that symbol in the market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.
        """
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
                    logger_main.log_and_print(
                        f"No market data available for symbol {symbol}. Skipping order processing.",
                        level="warning",
                    )
                    continue
                timeframe = min(available_timeframes)
                logger_main.log_and_print(
                    f"Order for {symbol} has no timeframe. Using lowest available: {timeframe}",
                    level="warning",
                )

            try:
                current_price = market_data[symbol][timeframe][3]  # Close price
            except KeyError:
                logger_main.log_and_print(
                    f"No market data for {symbol} at timeframe {timeframe}. Skipping order.",
                    level="warning",
                )
                continue

            logger_main.log_and_print(
                f"current_price : {current_price}",
                level="error",
            )

            is_filled, fill_price = order.is_filled(current_price)
            if is_filled:
                executed, trade = self.execute_order(order, fill_price)
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
        logger_main.log_and_print(
            f"Order {order.id} for {order.details.ticker} has expired and been canceled.",
            level="info",
        )

    def _update_position(
        self, symbol: str, position_change: Decimal, price: Decimal
    ) -> None:
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
        if self.margin_used:
            if equity / self.margin_used < self.margin_call_threshold:
                logger_main.log_and_print("Margin call triggered!", level="warning")
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

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return various performance metrics for the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics.
        """
        if not self.reporter:
            self.initialize_reporter()
        return self.reporter.generate_performance_summary()

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get the complete trade history.

        Returns:
            pd.DataFrame: A DataFrame containing the trade history.
        """
        if not self.reporter:
            self.initialize_reporter()
        return self.reporter.generate_trade_history_report()

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the equity curve data.
        """
        return self.metrics[["timestamp", "equity"]]

    def get_drawdown_curve(self) -> pd.DataFrame:
        """
        Get the drawdown curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the drawdown curve data.
        """
        if not self.reporter:
            self.initialize_reporter()
        equity_curve = self.get_equity_curve()
        equity_curve["Drawdown"] = self.reporter.calculate_max_drawdown()
        return equity_curve[["timestamp", "Drawdown"]]

    def get_position_history(self, symbol: str) -> pd.DataFrame:
        """
        Get the position history for a specific symbol.

        Args:
            symbol (str): The symbol to get the position history for.

        Returns:
            pd.DataFrame: A DataFrame containing the position history.
        """
        position_history = []
        for timestamp, equity in zip(self.metrics["timestamp"], self.metrics["equity"]):
            position_history.append(
                {"timestamp": timestamp, "size": self.positions.get(symbol, 0)}
            )
        return pd.DataFrame(position_history)

    def generate_report(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report using the Reporter.

        Args:
            start_date (Optional[datetime]): The start date for the report.
            end_date (Optional[datetime]): The end date for the report.

        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive report.
        """
        if not self.reporter:
            self.initialize_reporter()
        return self.reporter.generate_comprehensive_report(start_date, end_date)

    def plot_equity_curve(self) -> None:
        """
        Plot the equity curve using the Reporter.
        """
        if not self.reporter:
            self.initialize_reporter()
        self.reporter.plot_equity_curve()

    def plot_drawdown_curve(self) -> None:
        """
        Plot the drawdown curve using the Reporter.
        """
        if not self.reporter:
            self.initialize_reporter()
        self.reporter.plot_drawdown_curve()

    def plot_return_distribution(self) -> None:
        """
        Plot the return distribution using the Reporter.
        """
        if not self.reporter:
            self.initialize_reporter()
        self.reporter.plot_return_distribution()

    def export_report_to_csv(self, filename: str) -> None:
        """
        Export the trade history report to a CSV file using the Reporter.

        Args:
            filename (str): The name of the file to export to.
        """
        if not self.reporter:
            self.initialize_reporter()
        trade_history = self.reporter.generate_trade_history_report()
        self.reporter.export_to_csv(trade_history, filename)

    def export_report_to_excel(self, filename: str) -> None:
        """
        Export various reports to an Excel file using the Reporter.

        Args:
            filename (str): The name of the file to export to.
        """
        if not self.reporter:
            self.initialize_reporter()
        data = {
            "Performance Summary": pd.DataFrame(
                [self.reporter.generate_performance_summary()]
            ),
            "Trade History": self.reporter.generate_trade_history_report(),
            "Position Report": self.reporter.generate_position_report(),
            "Risk Report": pd.DataFrame([self.reporter.generate_risk_report()]),
        }
        self.reporter.export_to_excel(data, filename)

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
        return self.positions.get(symbol, Decimal("0"))

    def get_available_margin(self) -> Decimal:
        """
        Get the available margin for new trades.

        Returns:
            Decimal: The available margin.
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
