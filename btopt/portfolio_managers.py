import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data.bar import Bar
from .data.timeframe import Timeframe
from .order import Order, OrderDetails
from .trade import Trade
from .util.ext_decimal import ExtendedDecimal


class TradeManager:
    def __init__(self, commission_rate: ExtendedDecimal):
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.commission_rate = commission_rate
        self.trade_count = 0
        self.updated_trades: List[Trade] = []

    def create_trade(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Trade:
        self.trade_count += 1
        trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
            strategy_id=order.details.strategy_id,
        )
        trade.initial_size = order.get_filled_size()
        trade.current_size = trade.initial_size

        symbol = order.details.ticker
        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(trade)

        self._add_to_updated_trades(trade)
        return trade

    def update_trade(self, trade: Trade, current_bar: Bar) -> None:
        pre_update_state = trade.to_dict()
        trade.update(current_bar)
        if trade.to_dict() != pre_update_state:
            self._add_to_updated_trades(trade)

    def close_trade(
        self,
        trade: Trade,
        exit_order: Order,
        exit_price: ExtendedDecimal,
        exit_bar: Bar,
    ) -> None:
        trade.close(exit_order, exit_price, exit_bar)
        symbol = trade.ticker
        self.open_trades[symbol].remove(trade)
        if not self.open_trades[symbol]:
            del self.open_trades[symbol]
        self.closed_trades.append(trade)
        self._add_to_updated_trades(trade)

    def partial_close_trade(
        self,
        trade: Trade,
        exit_order: Order,
        exit_price: ExtendedDecimal,
        exit_bar: Bar,
        size: ExtendedDecimal,
    ) -> None:
        trade.close(exit_order, exit_price, exit_bar, size)
        self._add_to_updated_trades(trade)

    def get_open_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        if strategy_id:
            return [
                trade
                for trades in self.open_trades.values()
                for trade in trades
                if trade.strategy_id == strategy_id
            ]
        return [trade for trades in self.open_trades.values() for trade in trades]

    def get_closed_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        if strategy_id:
            return [
                trade
                for trade in self.closed_trades
                if trade.strategy_id == strategy_id
            ]
        return self.closed_trades

    def get_trades_for_symbol(self, symbol: str) -> List[Trade]:
        return self.open_trades.get(symbol, [])

    def get_position_size(self, symbol: str) -> ExtendedDecimal:
        return sum(trade.current_size for trade in self.open_trades.get(symbol, []))

    def calculate_unrealized_pnl(
        self, symbol: str, current_price: ExtendedDecimal
    ) -> ExtendedDecimal:
        return sum(
            trade.calculate_unrealized_pnl(current_price)
            for trade in self.open_trades.get(symbol, [])
        )

    def _add_to_updated_trades(self, trade: Trade) -> None:
        if trade not in self.updated_trades:
            self.updated_trades.append(trade)

    def get_updated_trades(self) -> List[Trade]:
        return self.updated_trades

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        return self.get_open_trades(strategy_id) + self.get_closed_trades(strategy_id)

    def clear_updated_trades(self) -> None:
        self.updated_trades.clear()


class AccountManager:
    def __init__(self, initial_capital: ExtendedDecimal, margin_ratio: ExtendedDecimal):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.margin_ratio = margin_ratio
        self.margin_used = ExtendedDecimal("0")
        self.equity = initial_capital
        self.buying_power = initial_capital
        self.unrealized_pnl = ExtendedDecimal("0")
        self.realized_pnl = ExtendedDecimal("0")
        self.transaction_log: List[Dict] = []

    def update_cash(self, amount: ExtendedDecimal, reason: str) -> None:
        self.cash += amount
        self._log_transaction("Cash", amount, reason)

    def update_margin(self, amount: ExtendedDecimal) -> None:
        self.margin_used += amount
        self._update_buying_power()

    def update_equity(self, unrealized_pnl: ExtendedDecimal) -> None:
        self.unrealized_pnl = unrealized_pnl
        self.equity = self.cash + self.unrealized_pnl
        self._update_buying_power()

    def realize_pnl(self, amount: ExtendedDecimal) -> None:
        self.realized_pnl += amount
        self.cash += amount
        self._log_transaction("PnL Realization", amount, "Trade closed")

    def _update_buying_power(self) -> None:
        self.buying_power = (self.equity - self.margin_used) / self.margin_ratio

    def check_margin_call(self, margin_call_threshold: ExtendedDecimal) -> bool:
        if self.margin_used > ExtendedDecimal("0"):
            return self.equity / self.margin_used < margin_call_threshold
        return False

    def get_account_summary(self) -> Dict[str, ExtendedDecimal]:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "buying_power": self.buying_power,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }

    def _log_transaction(
        self, transaction_type: str, amount: ExtendedDecimal, details: str
    ) -> None:
        self.transaction_log.append(
            {
                "timestamp": datetime.now(),
                "type": transaction_type,
                "amount": amount,
                "details": details,
            }
        )

    def get_transaction_log(self) -> List[Dict]:
        return self.transaction_log


class Position:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = ExtendedDecimal("0")
        self.average_price = ExtendedDecimal("0")
        self.unrealized_pnl = ExtendedDecimal("0")
        self.realized_pnl = ExtendedDecimal("0")


class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def update_position(self, order: Order, fill_price: ExtendedDecimal) -> None:
        symbol = order.details.ticker
        quantity = order.get_filled_size() * (
            1 if order.details.direction == Order.Direction.LONG else -1
        )

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)

        position = self.positions[symbol]

        if (
            position.quantity * quantity >= 0
        ):  # Adding to existing position or new position
            new_total = position.quantity + quantity
            if new_total != ExtendedDecimal("0"):
                position.average_price = (
                    position.average_price * position.quantity + fill_price * quantity
                ) / new_total
            position.quantity = new_total
        else:  # Reducing or closing position
            closed_quantity = min(abs(position.quantity), abs(quantity))
            pnl = (
                (fill_price - position.average_price)
                * closed_quantity
                * (-1 if position.quantity > 0 else 1)
            )
            position.realized_pnl += pnl
            position.quantity += quantity
            if position.quantity == ExtendedDecimal("0"):
                position.average_price = ExtendedDecimal("0")

        if position.quantity == ExtendedDecimal("0"):
            del self.positions[symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def calculate_position_value(
        self, symbol: str, current_price: ExtendedDecimal
    ) -> ExtendedDecimal:
        position = self.get_position(symbol)
        return position.quantity * current_price if position else ExtendedDecimal("0")

    def update_unrealized_pnl(
        self, symbol: str, current_price: ExtendedDecimal
    ) -> None:
        position = self.get_position(symbol)
        if position:
            position.unrealized_pnl = (
                current_price - position.average_price
            ) * position.quantity

    def get_long_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        return sum(
            position.quantity * current_prices[symbol]
            for symbol, position in self.positions.items()
            if position.quantity > 0
        )

    def get_short_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        return sum(
            abs(position.quantity) * current_prices[symbol]
            for symbol, position in self.positions.items()
            if position.quantity < 0
        )

    def get_all_positions(self) -> Dict[str, ExtendedDecimal]:
        return {
            symbol: position.quantity for symbol, position in self.positions.items()
        }

    def get_total_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        return sum(
            self.calculate_position_value(symbol, current_prices[symbol])
            for symbol in self.positions
        )

    def get_total_unrealized_pnl(self) -> ExtendedDecimal:
        return sum(position.unrealized_pnl for position in self.positions.values())

    def get_total_realized_pnl(self) -> ExtendedDecimal:
        return sum(position.realized_pnl for position in self.positions.values())

    def calculate_position_size(
        self,
        account_value: ExtendedDecimal,
        risk_per_trade: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        stop_loss: ExtendedDecimal,
    ) -> ExtendedDecimal:
        risk_amount = account_value * risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        return (
            risk_amount / price_difference
            if price_difference != ExtendedDecimal("0")
            else ExtendedDecimal("0")
        )

    def generate_close_all_orders(
        self,
        timestamp: datetime,
        timeframe: Timeframe,
        current_prices: Dict[str, ExtendedDecimal],
    ) -> List[Order]:
        """
        Generate market orders to close all open positions.

        Args:
            timestamp (datetime): The current timestamp for order creation.
            current_prices (Dict[str, ExtendedDecimal]): Current prices for all symbols.

        Returns:
            List[Order]: A list of market orders to close all positions.
        """
        close_orders = []
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                close_direction = (
                    Order.Direction.SHORT
                    if position.quantity > 0
                    else Order.Direction.LONG
                )
                order_details = OrderDetails(
                    ticker=symbol,
                    direction=close_direction,
                    size=abs(position.quantity),
                    price=current_prices[symbol],
                    exectype=Order.ExecType.MARKET,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    strategy_id="CLOSE_ALL",
                )
                close_orders.append(Order(str(uuid.uuid4()), order_details))

        return close_orders


class RiskManager:
    def __init__(
        self,
        initial_capital: ExtendedDecimal,
        max_position_size: ExtendedDecimal = ExtendedDecimal("1"),
        max_risk_per_trade: ExtendedDecimal = ExtendedDecimal("1"),
        max_risk_per_symbol: ExtendedDecimal = ExtendedDecimal("1"),
        max_drawdown: ExtendedDecimal = ExtendedDecimal("1"),
        var_confidence_level: float = 0.95,
        margin_ratio: ExtendedDecimal = ExtendedDecimal("0.01"),
        margin_call_threshold: ExtendedDecimal = ExtendedDecimal("0.01"),
    ):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_per_symbol = max_risk_per_symbol
        self.max_drawdown = max_drawdown
        self.var_confidence_level = var_confidence_level
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.equity_history: List[ExtendedDecimal] = [initial_capital]

    def calculate_position_size(
        self,
        account_value: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        stop_loss: ExtendedDecimal,
    ) -> ExtendedDecimal:
        risk_amount = account_value * self.max_risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        position_size = (
            risk_amount / price_difference
            if price_difference != ExtendedDecimal("0")
            else ExtendedDecimal("0")
        )
        return min(position_size, self.max_position_size)

    def check_risk_limits(
        self,
        order: Order,
        account_value: ExtendedDecimal,
        current_positions: Dict[str, ExtendedDecimal],
    ) -> bool:
        symbol = order.details.ticker
        new_position_size = (
            current_positions.get(symbol, ExtendedDecimal("0")) + order.details.size
        )

        # Check max position size
        if new_position_size > self.max_position_size:
            return False

        # Check max risk per symbol
        symbol_risk = (new_position_size * order.details.price) / account_value
        if symbol_risk > self.max_risk_per_symbol:
            return False

        return True

    def calculate_var(self, returns: pd.Series) -> ExtendedDecimal:
        return ExtendedDecimal(str(returns.quantile(1 - self.var_confidence_level)))

    def calculate_cvar(self, returns: pd.Series) -> ExtendedDecimal:
        var = self.calculate_var(returns)
        return ExtendedDecimal(str(returns[returns <= var].mean()))

    def calculate_drawdown(self, equity: ExtendedDecimal) -> ExtendedDecimal:
        self.equity_history.append(equity)
        peak = max(self.equity_history)
        return (peak - equity) / peak

    def check_drawdown(self, equity: ExtendedDecimal) -> bool:
        return self.calculate_drawdown(equity) <= self.max_drawdown

    def check_margin_call(
        self, equity: ExtendedDecimal, margin_used: ExtendedDecimal
    ) -> bool:
        if margin_used > ExtendedDecimal("0"):
            return equity / margin_used < self.margin_call_threshold
        return False

    def handle_margin_call(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
    ) -> List[Order]:
        """
        Handle a margin call by generating orders to close positions.

        Args:
            positions (Dict[str, ExtendedDecimal]): Current positions.
            current_prices (Dict[str, ExtendedDecimal]): Current market prices.

        Returns:
            List[Order]: List of orders to close positions to meet margin requirements.
        """
        orders_to_close = []
        sorted_positions = sorted(
            positions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for symbol, size in sorted_positions:
            close_size = abs(size)
            direction = Order.Direction.SHORT if size > 0 else Order.Direction.LONG

            order_details = OrderDetails(
                ticker=symbol,
                direction=direction,
                size=close_size,
                price=current_prices[symbol],
                exectype=Order.ExecType.MARKET,
                timestamp=datetime.now(),
                timeframe=None,  # This should be set appropriately
                strategy_id=None,  # This should be set appropriately
            )

            orders_to_close.append(Order(str(uuid.uuid4()), order_details))

            # Check if we've closed enough positions
            remaining_positions = {s: p for s, p in positions.items() if s != symbol}
            remaining_positions[symbol] = ExtendedDecimal("0")
            if not self.check_margin_call(
                self._calculate_equity(remaining_positions, current_prices),
                self._calculate_margin_used(remaining_positions, current_prices),
            ):
                break

        return orders_to_close

    def _calculate_equity(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
    ) -> ExtendedDecimal:
        return sum(
            abs(size) * current_prices[symbol] for symbol, size in positions.items()
        )

    def _calculate_margin_used(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
    ) -> ExtendedDecimal:
        return sum(
            abs(size) * current_prices[symbol] * self.margin_ratio
            for symbol, size in positions.items()
        )

    def get_risk_report(
        self, returns: pd.Series, equity: ExtendedDecimal
    ) -> Dict[str, ExtendedDecimal]:
        return {
            "VaR": self.calculate_var(returns),
            "CVaR": self.calculate_cvar(returns),
            "Drawdown": self.calculate_drawdown(equity),
            "Max Drawdown": max(
                self.calculate_drawdown(eq) for eq in self.equity_history
            ),
        }

    def select_position_to_reduce(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
        open_trades: List[Trade],
    ) -> Optional[Tuple[str, ExtendedDecimal]]:
        """
        Select a position to reduce during a margin call.

        This method uses a scoring system to determine which position to reduce.
        The scoring considers unrealized P&L, position size, and assumed market liquidity.

        Args:
            positions (Dict[str, ExtendedDecimal]): Current positions keyed by symbol.
            current_prices (Dict[str, ExtendedDecimal]): Current market prices keyed by symbol.
            open_trades (List[Trade]): List of all open trades.

        Returns:
            Optional[Tuple[str, ExtendedDecimal]]: A tuple containing the symbol to reduce and the amount to reduce by.
                                                   Returns None if no suitable position is found.
        """
        if not positions:
            return None

        position_scores = {}
        for symbol, quantity in positions.items():
            if quantity == 0:
                continue

            # Calculate unrealized P&L
            trades = [trade for trade in open_trades if trade.ticker == symbol]
            unrealized_pnl = sum(trade.get_unrealized_pnl() for trade in trades)

            # Calculate position value
            position_value = abs(quantity * current_prices[symbol])

            # Assume larger positions are in more liquid markets (this is a simplification)
            assumed_liquidity = abs(quantity)

            # Calculate score (lower is better to close)
            score = (
                unrealized_pnl / position_value if position_value != 0 else 0
            ) - assumed_liquidity / 1000

            position_scores[symbol] = (score, quantity)

        if not position_scores:
            return None

        # Select the position with the lowest score
        symbol_to_reduce = min(position_scores, key=lambda x: position_scores[x][0])

        # Determine the amount to reduce
        total_position = abs(positions[symbol_to_reduce])
        reduction_amount = min(
            total_position * ExtendedDecimal("0.2"),  # Reduce up to 20% of the position
            total_position,  # But not more than the total position
        )

        # Ensure we're not creating a very small leftover position
        if total_position - reduction_amount < total_position * ExtendedDecimal("0.1"):
            reduction_amount = total_position

        return symbol_to_reduce, reduction_amount

    def calculate_margin_excess_or_deficit(
        self, equity: ExtendedDecimal, margin_used: ExtendedDecimal
    ) -> ExtendedDecimal:
        """
        Calculate the excess margin or margin deficit.

        Args:
            equity (ExtendedDecimal): Current account equity.
            margin_used (ExtendedDecimal): Current margin used.

        Returns:
            ExtendedDecimal: The excess margin (positive) or margin deficit (negative).
        """
        required_margin = margin_used * self.margin_call_threshold
        return equity - required_margin

    def estimate_position_reduction_for_margin_call(
        self,
        equity: ExtendedDecimal,
        margin_used: ExtendedDecimal,
        position_value: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Estimate the position reduction needed to resolve a margin call.

        Args:
            equity (ExtendedDecimal): Current account equity.
            margin_used (ExtendedDecimal): Current margin used.
            position_value (ExtendedDecimal): Value of the position being considered for reduction.

        Returns:
            ExtendedDecimal: The estimated amount of position value to reduce.
        """
        margin_deficit = self.calculate_margin_excess_or_deficit(equity, margin_used)
        if margin_deficit >= 0:
            return ExtendedDecimal("0")

        # Calculate the reduction needed to bring the account to the margin call threshold
        reduction_needed = abs(margin_deficit) / (1 - self.margin_call_threshold)

        # Limit the reduction to the position value
        return min(reduction_needed, position_value)
