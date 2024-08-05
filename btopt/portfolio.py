from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .data.bar import Bar
from .data.dataview import DataView, DataViewNumpy
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails
from .trade import Trade


class Portfolio:
    """
    Represents a trading portfolio, managing trades, orders, and performance metrics.

    This class handles the execution of trades, tracking of open and closed positions,
    and calculation of various performance metrics.

    Attributes:
        initial_capital (Decimal): The starting capital of the portfolio.
        cash (Decimal): The current cash balance.
        commission_rate (Decimal): The commission rate for trades.
        spread (Decimal): The spread applied to trades.
        pyramiding (int): The maximum number of open trades allowed per symbol.
        open_trades (Dict[str, List[Trade]]): Currently open trades, organized by symbol.
        closed_trades (List[Trade]): List of closed trades.
        pending_orders (List[Order]): List of orders waiting to be executed.
        trade_count (int): Total number of trades executed.
        metrics (pd.DataFrame): DataFrame storing historical performance metrics.
        peak_equity (Decimal): The highest equity value reached.
        max_drawdown (Decimal): The maximum drawdown experienced.
    """

    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        commission_rate: Decimal = Decimal("0.001"),
        spread: Decimal = Decimal("0"),
        pyramiding: int = 1,
    ):
        """
        Initialize the Portfolio.

        Args:
            initial_capital (Decimal, optional): Starting capital. Defaults to 100,000.
            commission_rate (Decimal, optional): Commission rate for trades. Defaults to 0.1%.
            spread (Decimal, optional): Spread applied to trades. Defaults to 0.
            pyramiding (int, optional): Maximum open trades per symbol. Defaults to 1.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.spread = spread
        self.pyramiding = pyramiding
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0

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

    def update(
        self, timestamp: datetime, data_view: Union[DataView, DataViewNumpy]
    ) -> None:
        """
        Update the portfolio state based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            data_view (Union[DataView, DataViewNumpy]): The current market data view.
        """
        open_pnl = Decimal("0")
        commission = Decimal("0")
        slippage = Decimal("0")

        for symbol, trades in self.open_trades.items():
            current_bar = data_view.get_data_point(symbol, Timeframe("1m"), timestamp)
            if current_bar is None:
                continue

            for trade in trades:
                trade.update(current_bar)
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

    def execute_order(
        self, order: Order, execution_bar: Bar
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update the portfolio accordingly.

        Args:
            order (Order): The order to execute.
            execution_bar (Bar): The price bar at which the order is executed.

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

        execution_price = execution_bar.open  # Assume execution at the open price

        # Apply spread
        if order.details.direction == Order.Direction.LONG:
            execution_price += self.spread / 2
        else:
            execution_price -= self.spread / 2

        cost = execution_price * order.details.size
        commission = cost * self.commission_rate

        if order.details.direction == Order.Direction.LONG:
            if cost + commission > self.cash:
                logger_main.warning(f"Insufficient funds to execute order: {order}")
                return False, None
            self.cash -= cost + commission
        else:  # SHORT
            self.cash += cost - commission

        self.trade_count += 1
        trade = Trade(
            self.trade_count, order, execution_bar, commission_rate=self.commission_rate
        )

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(trade)

        logger_main.info(f"Executed order: {order}, resulting trade: {trade}")
        return True, trade

    def close_trade(self, trade: Trade, close_bar: Bar) -> None:
        """
        Close an existing trade.

        Args:
            trade (Trade): The trade to close.
            close_bar (Bar): The price bar at which the trade is closed.
        """
        close_price = close_bar.open  # Assume closing at the open price
        close_order = Order(
            order_id=hash(f"close_{trade.id}_{close_bar.timestamp}"),
            details=OrderDetails(
                ticker=trade.ticker,
                direction=Order.Direction.SHORT
                if trade.direction == Order.Direction.LONG
                else Order.Direction.LONG,
                size=trade.current_size,
                price=close_price,
                exectype=Order.ExecType.MARKET,
                timestamp=close_bar.timestamp,
                timeframe=trade.entry_bar.timeframe,
            ),
        )
        trade.close(close_order, close_bar)

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

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get the current position for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            Optional[Dict]: A dictionary containing position details, or None if no position exists.
        """
        if symbol not in self.open_trades:
            return None

        trades = self.open_trades[symbol]
        total_size = sum(trade.current_size for trade in trades)
        avg_entry_price = (
            sum(trade.entry_price * trade.current_size for trade in trades) / total_size
        )
        unrealized_pnl = sum(trade.metrics.pnl for trade in trades)

        return {
            "symbol": symbol,
            "size": total_size,
            "avg_entry_price": avg_entry_price,
            "unrealized_pnl": unrealized_pnl,
            "trades": trades,
        }

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

    def can_open_new_trade(self, order: Order) -> bool:
        """
        Check if a new trade can be opened based on the current portfolio state.

        Args:
            order (Order): The proposed order.

        Returns:
            bool: True if the trade can be opened, False otherwise.
        """
        symbol = order.details.ticker
        if (
            symbol in self.open_trades
            and len(self.open_trades[symbol]) >= self.pyramiding
        ):
            return False

        if order.details.direction == Order.Direction.LONG:
            cost = order.details.price * order.details.size
            commission = cost * self.commission_rate
            return cost + commission <= self.cash
        return True  # Assuming no margin requirements for short trades

    def generate_report(self) -> str:
        """
        Generate a comprehensive performance report.

        Returns:
            str: A formatted string containing the performance report.
        """
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.metrics.pnl > 0)
        losing_trades = total_trades - winning_trades

        total_profit = sum(
            trade.metrics.pnl for trade in self.closed_trades if trade.metrics.pnl > 0
        )
        total_loss = sum(
            trade.metrics.pnl for trade in self.closed_trades if trade.metrics.pnl < 0
        )

        avg_profit = (
            total_profit / winning_trades if winning_trades > 0 else Decimal("0")
        )
        avg_loss = total_loss / losing_trades if losing_trades > 0 else Decimal("0")

        win_rate = (
            (winning_trades / total_trades) * 100 if total_trades > 0 else Decimal("0")
        )
        profit_factor = (
            abs(total_profit / total_loss) if total_loss != 0 else Decimal("inf")
        )

        final_equity = self.calculate_equity()
        total_return = (
            (final_equity - self.initial_capital) / self.initial_capital
        ) * 100

        sharpe_ratio = self.calculate_sharpe_ratio()

        report = f"""
        Performance Report:
        ------------------
        Initial Capital: {self.initial_capital}
        Final Equity: {final_equity}
        Total Return: {total_return:.2f}%
        Max Drawdown: {self.max_drawdown:.2f}
        
        Total Trades: {total_trades}
        Winning Trades: {winning_trades}
        Losing Trades: {losing_trades}
        Win Rate: {win_rate:.2f}%
        
        Total Profit: {total_profit}
        Total Loss: {total_loss}
        Average Profit: {avg_profit}
        Average Loss: {avg_loss}
        Profit Factor: {profit_factor:.2f}
        
        Sharpe Ratio: {sharpe_ratio:.2f}
        """

        return report

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert portfolio history to a DataFrame for analysis.

        Returns:
            pd.DataFrame: A DataFrame containing the portfolio metrics over time.
        """
        return self.metrics

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.
        """
        self.__init__(
            self.initial_capital, self.commission_rate, self.spread, self.pyramiding
        )

    def get_open_trades_count(self) -> int:
        """
        Get the number of currently open trades.

        Returns:
            int: The number of open trades.
        """
        return sum(len(trades) for trades in self.open_trades.values())

    def get_total_exposure(self) -> Decimal:
        """
        Calculate the total exposure of the portfolio.

        Returns:
            Decimal: The total exposure as a percentage of the portfolio equity.
        """
        total_position_value = sum(
            abs(trade.current_size * trade.entry_price)
            for trades in self.open_trades.values()
            for trade in trades
        )
        return (total_position_value / self.calculate_equity()) * 100

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

    def process_pending_orders(
        self, data_view: Union[DataView, DataViewNumpy], timestamp: datetime
    ) -> List[Tuple[Order, bool, Optional[Trade]]]:
        """
        Process all pending orders based on the current market data.

        Args:
            data_view (Union[DataView, DataViewNumpy]): The current market data view.
            timestamp (datetime): The current timestamp.

        Returns:
            List[Tuple[Order, bool, Optional[Trade]]]: A list of tuples containing the order,
                                                    whether it was executed, and the resulting trade if any.
        """
        results = []
        for order in self.pending_orders[:]:  # Create a copy to iterate over
            current_bar = data_view.get_data_point(
                order.details.ticker, order.details.timeframe, timestamp
            )
            if current_bar is None:
                continue

            is_filled, fill_price = order.is_filled(current_bar)
            if is_filled:
                executed, trade = self.execute_order(order, current_bar)
                results.append((order, executed, trade))
                self.remove_pending_order(order)
            elif order.is_expired(timestamp):
                self.remove_pending_order(order)
                results.append((order, False, None))

        return results
