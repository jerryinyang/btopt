import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .data.bar import Bar
from .data.timeframe import Timeframe
from .order import BracketGroup, OCAGroup, OCOGroup, Order, OrderDetails, OrderGroup
from .trade import Trade
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


class OrderManager:
    """
    Manages the creation, modification, and cancellation of orders and order groups.
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_groups: Dict[str, OrderGroup] = {}
        self.pending_orders: Dict[str, List[Order]] = {}
        self.updated_orders: List[Order] = []

    def create_order(self, details: OrderDetails) -> Order:
        """
        Create a new order and add it to the pending orders.

        Args:
            details (OrderDetails): The details of the order to be created.

        Returns:
            Order: The newly created order.
        """
        order_id = str(uuid.uuid4())
        order = Order(order_id, details)
        self.orders[order_id] = order
        self.updated_orders.append(order)

        symbol = details.ticker
        if symbol not in self.pending_orders:
            self.pending_orders[symbol] = []
        self.pending_orders[symbol].append(order)

        logger_main.info(f"Created order: {order}")
        return order

    def create_oco_group(self, order1: Order, order2: Order) -> OCOGroup:
        """
        Create a new OCO (One-Cancels-the-Other) group.

        Args:
            order1 (Order): The first order in the OCO group.
            order2 (Order): The second order in the OCO group.

        Returns:
            OCOGroup: The newly created OCO group.
        """
        oco_group = OCOGroup()
        oco_group.add_order(order1)
        oco_group.add_order(order2)
        self.order_groups[oco_group.id] = oco_group
        logger_main.info(f"Created OCO group: {oco_group.id}")
        return oco_group

    def create_oca_group(self, orders: List[Order]) -> OCAGroup:
        """
        Create a new OCA (One-Cancels-All) group.

        Args:
            orders (List[Order]): The list of orders to be included in the OCA group.

        Returns:
            OCAGroup: The newly created OCA group.
        """
        oca_group = OCAGroup()
        for order in orders:
            oca_group.add_order(order)
        self.order_groups[oca_group.id] = oca_group
        logger_main.info(f"Created OCA group: {oca_group.id}")
        return oca_group

    def create_bracket_order(
        self,
        entry_order: Order,
        take_profit_order: Optional[Order] = None,
        stop_loss_order: Optional[Order] = None,
    ) -> BracketGroup:
        """
        Create a new Bracket order group.

        Args:
            entry_order (Order): The entry order for the bracket.
            take_profit_order (Optional[Order]): The take-profit order for the bracket. Defaults to None.
            stop_loss_order (Optional[Order]): The stop-loss order for the bracket. Defaults to None.

        Returns:
            BracketGroup: The newly created Bracket group.

        Raises:
            ValueError: If the entry order is not provided.
        """
        if not entry_order:
            logger_main.log_and_raise(
                ValueError("Entry order must be provided for a bracket order.")
            )

        if not take_profit_order and not stop_loss_order:
            logger_main.log_and_raise(
                ValueError(
                    "At least one out of the Take Profit order and the Stop Loss order must be provided for a bracket order."
                )
            )

        bracket_group = BracketGroup()
        bracket_group.add_order(entry_order, BracketGroup.Role.ENTRY)

        if take_profit_order:
            bracket_group.add_order(take_profit_order, BracketGroup.Role.LIMIT)

        if stop_loss_order:
            bracket_group.add_order(stop_loss_order, BracketGroup.Role.STOP)

        self.order_groups[bracket_group.id] = bracket_group
        logger_main.info(f"Created Bracket group: {bracket_group.id}")
        return bracket_group

    def update_order_group(self, group_id: str, status: str) -> None:
        """
        Update the status of an order group.

        Args:
            group_id (str): The ID of the order group to update.
            status (str): The new status of the order group.
        """
        if group_id in self.order_groups:
            group = self.order_groups[group_id]
            if status == "Active":
                group.activate()
            elif status == "Inactive":
                group.deactivate()
            logger_main.info(f"Updated order group {group_id} status to {status}")
        else:
            logger_main.warning(f"Order group not found: {group_id}")

    def get_order_group(self, group_id: str) -> Optional[OrderGroup]:
        """
        Get an order group by its ID.

        Args:
            group_id (str): The ID of the order group to retrieve.

        Returns:
            Optional[OrderGroup]: The order group if found, None otherwise.
        """
        return self.order_groups.get(group_id)

    def handle_order_update(self, order: Order) -> None:
        """
        Handle updates to an order, including its impact on the order group.

        Args:
            order (Order): The updated order.
        """
        self.updated_orders.append(order)
        if order.order_group:
            group = order.order_group
            if order.status == Order.Status.FILLED:
                group.on_order_filled(order)
            elif order.status == Order.Status.CANCELED:
                group.on_order_cancelled(order)
            elif order.status == Order.Status.REJECTED:
                group.on_order_rejected(order)

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an order.

        Args:
            order_id (str): The ID of the order to be cancelled.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.cancel()
            self.updated_orders.append(order)
            logger_main.info(f"Cancelled order: {order_id}")
        else:
            logger_main.warning(f"Order not found: {order_id}")

    def _sort_orders(self, orders: List[Order], bar: Bar) -> List[Order]:
        """
        Sort orders based on their type and price.

        Args:
            orders (List[Order]): The list of orders to sort.
            bar (Bar): The current price bar.

        Returns:
            List[Order]: The sorted list of orders.
        """
        is_up_bar = bar.close >= bar.open
        return sorted(orders, key=lambda x: x.sort_key(), reverse=not is_up_bar)

    def modify_order(self, order_id: str, new_details: Dict[str, Any]) -> None:
        """
        Modify an existing order.

        Args:
            order_id (str): The ID of the order to be modified.
            new_details (Dict[str, Any]): A dictionary containing the new details for the order.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            for key, value in new_details.items():
                if hasattr(order.details, key):
                    setattr(order.details, key, value)
            self.updated_orders.append(order)
            logger_main.info(f"Modified order: {order_id}")
        else:
            logger_main.warning(f"Order not found: {order_id}")

    def process_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> List[Order]:
        """
        Process all pending orders based on current market data.

        This method iterates through all pending orders, checks if they can be filled
        based on the current market data, and executes them if conditions are met.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.

        Returns:
            List[Order]: A list of orders that were processed (filled or cancelled).
        """
        processed_orders: List[Order] = []

        for symbol, orders in self.pending_orders.items():
            if symbol not in market_data:
                continue

            bar = market_data[symbol][min(market_data[symbol].keys())]
            sorted_orders = self._sort_orders(orders, bar)

            for order in sorted_orders:
                if not order.is_active:
                    continue

                is_filled, fill_price = order.is_filled(bar)

                if is_filled:
                    order.on_fill(order.get_remaining_size(), fill_price, timestamp)
                    processed_orders.append(order)
                    self.updated_orders.append(order)

                    if order.order_group:
                        order.order_group.on_order_filled(order)

                elif order.is_expired(timestamp):
                    order.on_cancel()
                    processed_orders.append(order)
                    self.updated_orders.append(order)

        # Remove processed orders from pending orders
        for order in processed_orders:
            self.pending_orders[order.details.ticker].remove(order)

        self.cleanup_completed_orders_and_groups()
        self.handle_order_group_events()  # Add this line to handle order group events

        return processed_orders

    def _get_current_price(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current price for a symbol. This method should be implemented
        to fetch the most recent price from the market data or a price feed.
        """
        # Implementation depends on how you're storing/accessing current market data
        pass

    def cancel_group(self, group_id: str) -> None:
        """
        Cancel all orders in a group.

        Args:
            group_id (str): The ID of the order group to be cancelled.
        """
        if group_id in self.order_groups:
            group = self.order_groups[group_id]
            for order in group.orders:
                order.cancel()
            logger_main.info(f"Cancelled order group: {group_id}")
        else:
            logger_main.warning(f"Order group not found: {group_id}")

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Retrieve an order by its ID.

        Args:
            order_id (str): The ID of the order to retrieve.

        Returns:
            Optional[Order]: The Order object if found, None otherwise.
        """
        return self.orders.get(order_id)

    def get_group(self, group_id: str) -> Optional[OrderGroup]:
        """
        Retrieve an order group by its ID.

        Args:
            group_id (str): The ID of the order group to retrieve.

        Returns:
            Optional[OrderGroup]: The OrderGroup object if found, None otherwise.
        """
        return self.order_groups.get(group_id)

    def get_active_orders(self) -> List[Order]:
        """
        Retrieve all active orders.

        Returns:
            List[Order]: A list of all active orders.
        """
        return [order for order in self.orders.values() if order.is_active]

    def get_active_groups(self) -> List[OrderGroup]:
        """
        Retrieve all active order groups.

        Returns:
            List[OrderGroup]: A list of all active order groups.
        """
        return [
            group
            for group in self.order_groups.values()
            if group.get_status() == "Active"
        ]

    def handle_fill_event(
        self,
        order_id: str,
        fill_size: ExtendedDecimal,
        fill_price: ExtendedDecimal,
        timestamp: datetime,
    ) -> None:
        """
        Handle a fill event for an order.

        Args:
            order_id (str): The ID of the order that was filled.
            fill_size (ExtendedDecimal): The size of the fill.
            fill_price (ExtendedDecimal): The price of the fill.
            timestamp (datetime): The timestamp of the fill.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.on_fill(fill_size, fill_price, timestamp)
            self.updated_orders.append(order)
        else:
            logger_main.warning(
                f"Fill event received for unknown order: {order_id}. \n\n ORDERS: {self.orders}\n ORDER GROUPS: {self.order_groups}\n\n"
            )

    def handle_reject_event(self, order_id: str, reason: str) -> None:
        """
        Handle a reject event for an order.

        Args:
            order_id (str): The ID of the order that was rejected.
            reason (str): The reason for the rejection.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.on_reject(reason)
            self.updated_orders.append(order)
        else:
            logger_main.warning(f"Reject event received for unknown order: {order_id}")

    def handle_order_group_events(self) -> None:
        """
        Handle events for order groups.

        This method checks the status of each order group and removes
        completed groups from the active order_groups dictionary.
        """
        for group_id, group in list(self.order_groups.items()):
            if group.get_status() in ["Filled", "Cancelled/Rejected"]:
                del self.order_groups[group_id]
                logger_main.info(f"Order group {group_id} completed and removed.")

    def cleanup_completed_orders_and_groups(self) -> None:
        """
        Remove completed orders and groups from the manager.
        """
        self.orders = {
            order_id: order
            for order_id, order in self.orders.items()
            if order.is_active
        }
        self.order_groups = {
            group_id: group
            for group_id, group in self.order_groups.items()
            if group.get_status() == "Active"
        }
        logger_main.info("Cleaned up completed orders and groups")

    def check_order_expiry(self, current_time: datetime) -> None:
        """
        Check and cancel expired orders.

        Args:
            current_time (datetime): The current timestamp to check against.
        """
        for order_id, order in list(self.orders.items()):
            if order.is_expired(current_time):
                self.cancel_order(order_id)
                logger_main.info(f"Order {order_id} expired and cancelled.")

    def get_updated_orders(self) -> List[Order]:
        return self.updated_orders

    def clear_updated_orders(self) -> None:
        self.updated_orders.clear()

    def __repr__(self) -> str:
        """Return a string representation of the OrderManager."""
        return f"OrderManager(active_orders={len(self.get_active_orders())}, active_groups={len(self.get_active_groups())})"


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
        """
        Close a trade.

        Args:
            trade (Trade): The trade to close.
            exit_order (Order): The order used to close the trade.
            exit_price (ExtendedDecimal): The price at which the trade is being closed.
            exit_bar (Bar): The bar at which the trade is being closed.
        """
        trade.close(exit_order, exit_price, exit_bar)
        symbol = trade.ticker
        if trade.status == Trade.Status.CLOSED:
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
        """
        Partially close a trade.

        Args:
            trade (Trade): The trade to partially close.
            exit_order (Order): The order used to close part of the trade.
            exit_price (ExtendedDecimal): The price at which part of the trade is being closed.
            exit_bar (Bar): The bar at which part of the trade is being closed.
            size (ExtendedDecimal): The size of the position to close.
        """
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
        max_risk: ExtendedDecimal = ExtendedDecimal("1"),
        max_risk_per_trade: ExtendedDecimal = ExtendedDecimal("1"),
        max_risk_per_symbol: ExtendedDecimal = ExtendedDecimal("1"),
        max_drawdown: ExtendedDecimal = ExtendedDecimal("0.9"),
        var_confidence_level: float = 0.95,
        margin_ratio: ExtendedDecimal = ExtendedDecimal("0.01"),
        margin_call_threshold: ExtendedDecimal = ExtendedDecimal("0.01"),
        symbols: List[str] = [],
    ):
        self.initial_capital = initial_capital
        self.max_risk = max_risk
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_per_symbol = max_risk_per_symbol
        self.max_drawdown = max_drawdown
        self.var_confidence_level = var_confidence_level
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.equity_history: List[ExtendedDecimal] = [initial_capital]
        self.symbol_weights: Dict[str, ExtendedDecimal] = {}
        self._initialize_symbol_weights(symbols)

    def _initialize_symbol_weights(self, symbols: List[str]) -> None:
        """Initialize symbol weights equally among all symbols."""
        if symbols:
            weight = ExtendedDecimal("1") / ExtendedDecimal(str(len(symbols)))
            self.symbol_weights = {symbol: weight for symbol in symbols}

    def set_symbol_weights(self, weights: Dict[str, ExtendedDecimal]) -> None:
        """Set and normalize symbol weights."""
        total_weight = sum(weights.values())
        if total_weight > ExtendedDecimal("1"):
            normalized_weights = {
                symbol: weight / total_weight for symbol, weight in weights.items()
            }
        else:
            normalized_weights = weights

        for symbol, weight in normalized_weights.items():
            if weight > self.max_risk_per_symbol:
                normalized_weights[symbol] = self.max_risk_per_symbol

        self.symbol_weights = normalized_weights

    def get_symbol_weight(self, symbol: str) -> ExtendedDecimal:
        """Get the weight for a specific symbol."""
        return self.symbol_weights.get(symbol, ExtendedDecimal("0"))

    def get_all_symbol_weights(self) -> Dict[str, ExtendedDecimal]:
        """Get all symbol weights."""
        return self.symbol_weights.copy()

    def set_all_symbol_weights(self, weights: Dict[str, ExtendedDecimal]) -> None:
        """Set weights for all symbols."""
        self.set_symbol_weights(weights)

    def calculate_risk_amount(
        self, symbol: str, account_value: ExtendedDecimal
    ) -> ExtendedDecimal:
        """Calculate the risk amount for a specific symbol."""
        symbol_weight = self.get_symbol_weight(symbol)
        risk_amount = account_value * self.max_risk * symbol_weight
        return min(risk_amount, account_value * self.max_risk_per_trade)

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
