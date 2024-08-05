from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data.bar import Bar
from .log_config import logger_main
from .order import Order, OrderDetails
from .trade import Trade


class Portfolio:
    def __init__(
        self, initial_capital: Decimal, commission_rate: Decimal = Decimal("0.001")
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0

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

    def update(self, timestamp: datetime, current_prices: Dict[str, Decimal]) -> None:
        open_pnl = Decimal("0")
        commission = Decimal("0")
        slippage = Decimal("0")

        for symbol, trades in self.open_trades.items():
            if symbol not in current_prices:
                continue
            current_price = current_prices[symbol]

            for trade in trades:
                trade.update(
                    Bar(
                        close=current_price,
                        timestamp=timestamp,
                        timeframe=trade.entry_bar.timeframe,
                        ticker=symbol,
                    )
                )
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
        self, order: Order, symbol: str, price: Decimal, timestamp: datetime
    ) -> Tuple[bool, Optional[Trade]]:
        cost = price * order.details.size
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
            self.trade_count,
            order,
            Bar(
                close=price,
                timestamp=timestamp,
                timeframe=order.details.timeframe,
                ticker=symbol,
            ),
            commission_rate=self.commission_rate,
        )

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(trade)

        logger_main.info(f"Executed order: {order}, resulting trade: {trade}")
        return True, trade

    def close_trade(self, trade: Trade, price: Decimal, timestamp: datetime) -> None:
        close_order = Order(
            order_id=hash(f"close_{trade.id}_{timestamp}"),
            details=OrderDetails(
                ticker=trade.ticker,
                direction=Order.Direction.SHORT
                if trade.direction == Order.Direction.LONG
                else Order.Direction.LONG,
                size=trade.current_size,
                price=price,
                exectype=Order.ExecType.MARKET,
                timestamp=timestamp,
                timeframe=trade.entry_bar.timeframe,
            ),
        )
        trade.close(
            close_order,
            Bar(
                close=price,
                timestamp=timestamp,
                timeframe=trade.entry_bar.timeframe,
                ticker=trade.ticker,
            ),
        )

        symbol = trade.ticker
        self.open_trades[symbol].remove(trade)
        if not self.open_trades[symbol]:
            del self.open_trades[symbol]
        self.closed_trades.append(trade)

        close_cost = price * trade.current_size
        commission = close_cost * self.commission_rate

        if trade.direction == Order.Direction.LONG:
            self.cash += close_cost - commission
        else:  # SHORT
            self.cash -= close_cost + commission

        logger_main.info(f"Closed trade: {trade}")

    def get_results(self) -> Dict:
        final_equity = self.calculate_equity()
        total_return = (
            (final_equity - self.initial_capital) / self.initial_capital
        ) * 100
        sharpe_ratio = self.calculate_sharpe_ratio()

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(self.closed_trades),
            "win_rate": self.calculate_win_rate(),
            "profit_factor": self.calculate_profit_factor(),
        }

    def calculate_equity(self) -> Decimal:
        open_pnl = sum(
            trade.metrics.pnl
            for trades in self.open_trades.values()
            for trade in trades
        )
        closed_pnl = sum(trade.metrics.pnl for trade in self.closed_trades)
        return self.cash + open_pnl + closed_pnl

    def calculate_sharpe_ratio(
        self, risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal:
        if len(self.metrics) < 2:
            return Decimal("0")

        returns = self.metrics["equity"].pct_change().dropna()
        excess_returns = returns - (
            risk_free_rate / 252
        )  # Assuming daily returns and 252 trading days
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        return Decimal(str(sharpe_ratio))

    def calculate_win_rate(self) -> Decimal:
        if not self.closed_trades:
            return Decimal("0")
        winning_trades = sum(1 for trade in self.closed_trades if trade.metrics.pnl > 0)
        return Decimal(str(winning_trades / len(self.closed_trades) * 100))

    def calculate_profit_factor(self) -> Decimal:
        total_profit = sum(
            trade.metrics.pnl for trade in self.closed_trades if trade.metrics.pnl > 0
        )
        total_loss = sum(
            abs(trade.metrics.pnl)
            for trade in self.closed_trades
            if trade.metrics.pnl < 0
        )
        return total_profit / total_loss if total_loss != 0 else Decimal("inf")

    def reset(self) -> None:
        self.__init__(self.initial_capital, self.commission_rate)

    def get_open_trades_count(self) -> int:
        return sum(len(trades) for trades in self.open_trades.values())

    def get_total_exposure(self) -> Decimal:
        total_position_value = sum(
            abs(trade.current_size * trade.entry_price)
            for trades in self.open_trades.values()
            for trade in trades
        )
        return (total_position_value / self.calculate_equity()) * 100

    def add_pending_order(self, order: Order) -> None:
        self.pending_orders.append(order)
        logger_main.info(f"Added pending order: {order}")

    def remove_pending_order(self, order: Order) -> None:
        if order in self.pending_orders:
            self.pending_orders.remove(order)
            logger_main.info(f"Removed pending order: {order}")
        else:
            logger_main.warning(
                f"Attempted to remove non-existent pending order: {order}"
            )

    def process_pending_orders(
        self, current_prices: Dict[str, Decimal], timestamp: datetime
    ) -> List[Tuple[Order, bool, Optional[Trade]]]:
        results = []
        for order in self.pending_orders[:]:
            if order.details.ticker not in current_prices:
                continue

            current_price = current_prices[order.details.ticker]
            is_filled, fill_price = order.is_filled(
                Bar(
                    close=current_price,
                    timestamp=timestamp,
                    timeframe=order.details.timeframe,
                    ticker=order.details.ticker,
                )
            )
            if is_filled:
                executed, trade = self.execute_order(
                    order, order.details.ticker, fill_price, timestamp
                )
                results.append((order, executed, trade))
                self.remove_pending_order(order)
            elif order.is_expired(timestamp):
                self.remove_pending_order(order)
                results.append((order, False, None))

        return results

    def get_portfolio_status(self) -> Dict:
        return {
            "cash": self.cash,
            "equity": self.calculate_equity(),
            "open_trades_count": self.get_open_trades_count(),
            "total_exposure": self.get_total_exposure(),
            "max_drawdown": self.max_drawdown,
        }
