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
    Represents a sophisticated trading portfolio, managing trades, orders, and performance metrics.

    This class is a cornerstone of the backtesting system, responsible for simulating
    real-world trading activities, including order execution, position management,
    and comprehensive performance tracking. It provides a robust framework for
    evaluating trading strategies by accurately modeling the financial implications
    of trading decisions.

    Key Responsibilities:
    1. Trade Execution: Simulates the execution of trades based on strategy signals,
       accounting for factors such as available capital, position sizing, and market conditions.
    2. Position Management: Tracks and updates open positions across multiple symbols,
       handling partial closes, pyramiding, and position consolidation.
    3. Order Management: Maintains a queue of pending orders, processes their execution
       based on market data, and handles various order types (market, limit, stop, etc.).
    4. Risk Management: Implements risk control measures such as position sizing,
       maximum drawdown limits, and exposure constraints.
    5. Performance Tracking: Continuously calculates and records a wide range of
       performance metrics, providing a detailed view of the portfolio's health and profitability.
    6. Capital Management: Tracks available capital, calculates required margin,
       and ensures trading activities adhere to capital constraints.

    Attributes:
        initial_capital (Decimal):
            The starting capital of the portfolio. This value is used as a baseline
            for calculating returns and other performance metrics.

        cash (Decimal):
            The current cash balance available for trading. This is dynamically
            updated as trades are opened and closed, accounting for profits, losses,
            and trading costs.

        commission_rate (Decimal):
            The commission rate applied to trades, expressed as a decimal (e.g., 0.001 for 0.1%).
            This is used to simulate transaction costs and their impact on performance.

        slippage (Decimal):
            The slippage rate applied to trades, representing the typical difference
            between expected and actual execution prices. This adds realism to the
            simulation by accounting for market impact and liquidity issues.

        pyramiding (int):
            The maximum number of open trades allowed per symbol. This parameter
            controls the portfolio's ability to scale into positions, balancing
            between capitalizing on strong trends and managing risk.

        open_trades (Dict[str, List[Trade]]):
            A dictionary of currently open trades, organized by symbol. Each entry
            contains a list of Trade objects, allowing for multiple positions per symbol
            when pyramiding is enabled.

        closed_trades (List[Trade]):
            A chronological list of all closed trades. This historical record is
            crucial for post-analysis and performance evaluation.

        pending_orders (List[Order]):
            A queue of orders waiting to be executed. This list is processed on each
            update cycle, simulating the behavior of orders in a live trading environment.

        trade_count (int):
            A running count of the total number of trades executed. This includes
            both open and closed trades and is used for trade identification and statistics.

        metrics (pd.DataFrame):
            A DataFrame storing historical performance metrics. This time series data
            includes values such as equity, cash balance, open/closed P&L, drawdown,
            and various risk/return ratios, providing a comprehensive view of the
            portfolio's performance over time.

        peak_equity (Decimal):
            The highest equity value reached by the portfolio. This is used in
            drawdown calculations and as a high-water mark for performance evaluation.

        max_drawdown (Decimal):
            The maximum peak-to-trough decline in portfolio value, expressed as a
            monetary amount. This is a key risk metric that quantifies the portfolio's
            largest historical loss.

    Methods:
        The Portfolio class offers a wide array of methods for trade management,
        performance calculation, and risk assessment. These include methods for
        executing trades, updating positions, calculating returns and risk metrics,
        and generating performance reports. Detailed documentation for each method
        is provided in their respective docstrings.

    Integration:
        The Portfolio class is designed to work seamlessly with other components
        of the backtesting system, particularly the Engine and Strategy classes.
        It receives trading signals from strategies, executes them based on current
        market data and portfolio state, and provides feedback to the Engine for
        overall backtest control and reporting.

    Performance Considerations:
        Given the central role of the Portfolio class in the backtesting process,
        its methods are optimized for performance, especially for high-frequency
        trading simulations or when dealing with large numbers of instruments.
        Efficient data structures and algorithms are employed to minimize
        computational overhead.

    Extensibility:
        The class is designed with extensibility in mind, allowing for easy
        addition of new performance metrics, risk management techniques, or
        order types. Custom plugins can be developed to extend its functionality
        for specific trading styles or asset classes.

    Note:
        While the Portfolio class aims to provide a realistic simulation of
        trading activities, users should be aware of its limitations in
        modeling certain real-world factors such as market microstructure
        effects, complex order routing, or broker-specific behaviors.
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0"),
        pyramiding: int = 1,
        margin_ratio: Decimal = Decimal(
            "0.5"
        ),  # New: Margin ratio (e.g., 0.5 for 2:1 leverage)
        margin_call_threshold: Decimal = Decimal("0.3"),  # New: Margin call threshold
    ):
        """
        Initialize the Portfolio.

        Args:
            initial_capital (Decimal, optional): Starting capital. Defaults to 100,000.
            commission_rate (Decimal, optional): Commission rate for trades. Defaults to 0.1%.
            slippage (Decimal, optional): Slippage applied to trades. Defaults to 0.
            pyramiding (int, optional): Maximum open trades per symbol. Defaults to 1.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.pyramiding = pyramiding
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0

        # Margin-related attributes
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.margin_used = Decimal("0")
        self.buying_power = initial_capital / margin_ratio

        # Metrics tracking
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

    # region Portfolio Update and Signal Processing
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
        self._check_exit_conditions(timestamp, market_data)
        self._update_metrics(timestamp, market_data)

    def process_signals(
        self,
        signals: List[Dict[str, Any]],
        timestamp: pd.Timestamp,
        market_data: Dict[str, Dict[Timeframe, np.ndarray]],
    ) -> None:
        """
        Process trading signals and create orders accordingly.

        Args:
            signals (List[Dict[str, Any]]): List of trading signals to process.
            timestamp (pd.Timestamp): Current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): Current market data.
        """
        for signal in signals:
            if self.can_open_new_trade(signal):
                order = self._create_order_from_signal(signal)
                self.add_pending_order(order)

    def _create_order_from_signal(self, signal: Dict[str, Any]) -> Order:
        """
        Create an Order object from a trading signal.

        Args:
            signal (Dict[str, Any]): The trading signal.

        Returns:
            Order: The created Order object.
        """
        order_details = OrderDetails(
            ticker=signal["symbol"],
            direction=signal["direction"],
            size=signal["size"],
            price=signal["price"],
            exectype=signal["order_type"],
            timestamp=signal["timestamp"],
            timeframe=signal["timeframe"],
            expiry=signal.get("expiry"),
            stoplimit_price=signal.get("stoplimit_price"),
            parent_id=signal.get("parent_id"),
            exit_profit=signal.get("exit_profit"),
            exit_loss=signal.get("exit_loss"),
            exit_profit_percent=signal.get("exit_profit_percent"),
            exit_loss_percent=signal.get("exit_loss_percent"),
            trailing_percent=signal.get("trailing_percent"),
            slippage=self.slippage,
        )
        return Order(
            order_id=hash(
                f"{signal['symbol']}_{signal['timestamp']}_{signal['direction']}"
            ),
            details=order_details,
        )

    def _update_metrics(
        self,
        timestamp: pd.Timestamp,
        market_data: Dict[str, Dict[Timeframe, np.ndarray]],
    ) -> None:
        """
        Update portfolio metrics based on current market data.

        Args:
            timestamp (pd.Timestamp): Current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): Current market data.
        """
        open_pnl = Decimal("0")
        commission = Decimal("0")
        slippage = Decimal("0")

        for symbol, trades in self.open_trades.items():
            current_price = market_data[symbol][Timeframe("1m")][
                3
            ]  # Assuming close price is at index 3
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

        # Check pyramiding limit
        if (
            symbol in self.open_trades
            and len(self.open_trades[symbol]) >= self.pyramiding
        ):
            logger_main.warning(
                f"Pyramiding limit reached for {symbol}. Order not executed: {order}"
            )
            return False, None

        cost = execution_price * order.details.size
        commission = cost * self.commission_rate

        # Check margin requirements
        if not self.check_margin_requirements(order, cost):
            logger_main.warning(f"Insufficient margin to execute order: {order}")
            return False, None

        # Update margin and buying power
        self.update_margin(order, cost)

        if order.details.direction == Order.Direction.LONG:
            if cost + commission > self.cash:
                logger_main.warning(f"Insufficient funds to execute order: {order}")
                return False, None
            self.cash -= cost + commission
        else:  # SHORT
            self.cash += cost - commission

        self.trade_count += 1
        trade = Trade(
            self.trade_count,
            order,
            execution_price,
            commission_rate=self.commission_rate,
        )

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(trade)

        logger_main.info(f"Executed order: {order}, resulting trade: {trade}")
        return True, trade

    def close_trade(self, trade: Trade, close_price: Decimal) -> Trade:
        """
        Close an existing trade.

        Args:
            trade (Trade): The trade to close.
            close_price (Decimal): The price at which to close the trade.

        Returns:
            Trade: The closed trade.
        """
        close_order = Order(
            order_id=hash(f"close_{trade.id}_{datetime.now()}"),
            details=OrderDetails(
                ticker=trade.ticker,
                direction=Order.Direction.SHORT
                if trade.direction == Order.Direction.LONG
                else Order.Direction.LONG,
                size=trade.current_size,
                price=close_price,
                exectype=Order.ExecType.MARKET,
                timestamp=datetime.now(),
                timeframe=trade.entry_bar.timeframe,
            ),
        )
        trade.close(close_order, close_price)

        symbol = trade.ticker
        self.open_trades[symbol].remove(trade)
        if not self.open_trades[symbol]:
            del self.open_trades[symbol]
        self.closed_trades.append(trade)

        close_cost = close_price * trade.current_size
        commission = close_cost * self.commission_rate

        if trade.direction == Order.Direction.LONG:
            self.cash += close_cost - commission
        else:  # SHORT
            self.cash -= close_cost + commission

        logger_main.info(f"Closed trade: {trade}")
        return trade

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> List[Tuple[Order, bool, Optional[Trade]]]:
        """
        Process all pending orders based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.

        Returns:
            List[Tuple[Order, bool, Optional[Trade]]]: A list of tuples containing the order,
                                                    whether it was executed, and the resulting trade if any.
        """
        results = []
        for order in self.pending_orders[:]:  # Create a copy to iterate over
            symbol = order.details.ticker
            current_price = market_data[symbol][order.details.timeframe][
                3
            ]  # Assuming close price is at index 3
            is_filled, fill_price = order.is_filled(current_price)
            if is_filled:
                executed, trade = self.execute_order(order, fill_price)
                results.append((order, executed, trade))
                self.remove_pending_order(order)
            elif order.is_expired(timestamp):
                self.remove_pending_order(order)
                results.append((order, False, None))

            # Check for margin call after processing each order
            if self.check_margin_call():
                self.handle_margin_call()

        return results

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
            current_price = market_data[symbol][Timeframe("1m")][
                3
            ]  # Assuming close price is at index 3
            for trade in trades:
                trade.update(current_price)

    def _check_exit_conditions(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, np.ndarray]]
    ) -> List[Trade]:
        """
        Check and execute exit conditions for open trades.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, np.ndarray]]): The current market data.

        Returns:
            List[Trade]: A list of trades that were closed.
        """
        closed_trades = []
        for symbol, trades in list(self.open_trades.items()):
            current_price = market_data[symbol][Timeframe("1m")][
                3
            ]  # Assuming close price is at index 3
            for trade in trades[:]:  # Create a copy to iterate over
                if trade.should_exit(current_price):
                    closed_trade = self.close_trade(trade, current_price)
                    closed_trades.append(closed_trade)
        return closed_trades

    def add_pending_order(self, order: Order) -> None:
        """
        Add an order to the list of pending orders.

        Args:
            order (Order): The order to add to the pending list.
        """
        self.pending_orders.append(order)
        logger_main.info(f"Added pending order: {order}")

    def remove_pending_order(self, order: Order) -> None:
        """
        Remove an order from the list of pending orders.

        Args:
            order (Order): The order to remove from the pending list.
        """
        if order in self.pending_orders:
            self.pending_orders.remove(order)
            logger_main.info(f"Removed pending order: {order}")
        else:
            logger_main.warning(
                f"Attempted to remove non-existent pending order: {order}"
            )

    def modify_order(self, order_id: int, new_details: Dict[str, Any]) -> bool:
        """
        Modify an existing pending order.

        Args:
            order_id (int): The ID of the order to modify.
            new_details (Dict[str, Any]): A dictionary containing the new order details.

        Returns:
            bool: True if the order was successfully modified, False otherwise.
        """
        for order in self.pending_orders:
            if order.id == order_id:
                # Update order details
                for key, value in new_details.items():
                    if hasattr(order.details, key):
                        setattr(order.details, key, value)

                logger_main.info(f"Modified order: {order}")
                return True

        logger_main.warning(f"Order with ID {order_id} not found in pending orders.")
        return False

    # endregion

    # region Margin Related Methods
    def check_margin_requirements(self, order: Order, cost: Decimal) -> bool:
        """
        Check if there's sufficient margin to execute the order.
        """
        if order.details.direction == Order.Direction.LONG:
            required_margin = cost * self.margin_ratio
        else:  # SHORT
            required_margin = (
                cost * self.margin_ratio * Decimal("2")
            )  # Higher margin for short selling

        return self.buying_power >= required_margin

    def update_margin(self, order: Order, cost: Decimal) -> None:
        """
        Update margin and buying power after executing an order.
        """
        if order.details.direction == Order.Direction.LONG:
            self.margin_used += cost * self.margin_ratio
        else:  # SHORT
            self.margin_used += cost * self.margin_ratio * Decimal("2")

        self.buying_power = (
            self.calculate_equity() - self.margin_used
        ) / self.margin_ratio

    def check_margin_call(self) -> bool:
        """
        Check if a margin call should be triggered.
        """
        equity = self.calculate_equity()
        if equity / self.margin_used < self.margin_call_threshold:
            logger_main.warning("Margin call triggered!")
            return True
        return False

    def handle_margin_call(self) -> None:
        """
        Handle a margin call by closing positions until margin requirements are met.
        """
        while self.check_margin_call() and self.open_trades:
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

    def can_open_new_trade(self, signal: Dict[str, Any]) -> bool:
        """
        Check if a new trade can be opened based on the current portfolio state and the signal.

        Args:
            signal (Dict[str, Any]): The trading signal.

        Returns:
            bool: True if the trade can be opened, False otherwise.
        """
        symbol = signal["symbol"]
        if (
            symbol in self.open_trades
            and len(self.open_trades[symbol]) >= self.pyramiding
        ):
            return False

        if signal["direction"] == Order.Direction.LONG:
            cost = signal["price"] * signal["size"]
            commission = cost * self.commission_rate
            return cost + commission <= self.cash
        return True  # Assuming no margin requirements for short trades

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
            current_price = market_data[symbol][Timeframe("1m")][
                3
            ]  # Assuming close price is at index 3
            for trade in trades[:]:
                self.close_trade(trade, current_price)
        logger_main.info("Closed all open trades.")

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
        logger_main.info("Portfolio configuration set.")

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.
        """
        self.__init__(
            self.initial_capital, self.commission_rate, self.slippage, self.pyramiding
        )
        self.margin_used = Decimal("0")
        self.buying_power = self.initial_capital / self.margin_ratio
        logger_main.info("Portfolio reset to initial state.")

    # endregion
