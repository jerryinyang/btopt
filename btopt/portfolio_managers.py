import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

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
        self.order_groups: Dict[str, Union[OCOGroup, OCAGroup, BracketGroup]] = {}
        self.updated_orders: List[Order] = []

    def __repr__(self) -> str:
        """Return a string representation of the OrderManager."""
        return f"OrderManager(active_orders={len(self.get_active_orders())}, active_groups={len(self.get_active_groups())})"

    def create_order(self, details: OrderDetails, activated: bool = True) -> Order:
        """
        Create a new order and add it to the pending orders.

        Args:
            details (OrderDetails): The details of the order to be created.
            activated (bool): Controls if the order should be activated or deactivated

        Returns:
            Order: The newly created order.
        """
        order_id = str(uuid.uuid4())
        order = Order(order_id, details)

        if not activated:
            order.deactivate()

        # Map order to order_id
        self.orders[order_id] = order

        # Track updated orders
        self.updated_orders.append(order)
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
        oco_group.add_manager(self)

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
        oca_group.add_manager(self)

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

        if not (take_profit_order or stop_loss_order):
            logger_main.log_and_raise(
                ValueError(
                    "At least one out of the Take Profit order and the Stop Loss order must be provided for a bracket order."
                )
            )

        bracket_group = BracketGroup()
        bracket_group.add_manager(self)

        bracket_group.add_order(entry_order, BracketGroup.Role.ENTRY)

        if take_profit_order:
            bracket_group.add_order(take_profit_order, BracketGroup.Role.LIMIT)

        if stop_loss_order:
            bracket_group.add_order(stop_loss_order, BracketGroup.Role.STOP)

        self.order_groups[bracket_group.id] = bracket_group
        logger_main.info(f"Created Bracket group: {bracket_group.id}")
        return bracket_group

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id (str): The ID of the order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        order.cancel()
        self._handle_order_group(order)
        self.updated_orders.append(order)
        return True

    def process_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> List[Order]:
        """
        Process all pending orders based on current market data, handling partial fills.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.

        Returns:
            List[Order]: A list of orders that were processed (filled or cancelled).
        """
        processed_orders: List[Order] = []

        for symbol in market_data:
            bar = market_data[symbol][min(market_data[symbol].keys())]
            orders = [
                order
                for order in self.orders.values()
                if order.details.ticker == symbol
            ]
            sorted_orders = self._sort_orders(orders, bar)

            for order in sorted_orders:
                if not order.is_active:
                    continue

                is_filled, fill_price, fill_size = self._check_order_fill(order, bar)

                if is_filled:
                    self._handle_order_fill(order, fill_price, fill_size, timestamp)
                    processed_orders.append(order)

                elif order.is_expired(timestamp):
                    self._cancel_order(order)
                    processed_orders.append(order)

        self._cleanup_completed_orders()
        return processed_orders

    def _sort_orders(self, orders: List[Order], bar: Bar) -> List[Order]:
        """
        Sort a list of orders based on their execution type and the current market conditions.

        This method implements a custom sorting logic for orders:
        1. Market orders are always placed at the beginning of the list.
        2. Stop orders come after market orders but before limit orders.
        3. Limit orders are placed at the end of the list.
        4. For both stop and limit orders, the sorting is based on the current bar:
           - In an up bar (close >= open), orders are sorted in ascending price order.
           - In a down bar (close < open), orders are sorted in descending price order.

        The assumption is made that bar formation follows the OLHC (Open, Low, High, Close) path.

        Args:
            orders (List[Order]): A list of Order objects to be sorted.
            bar (Bar): The current price bar, used to determine market direction.

        Returns:
            List[Order]: A new list containing the input orders sorted according to the specified logic.
        """
        is_up_bar = bar.close >= bar.open

        # Separate orders into categories
        market_orders = [
            order for order in orders if order.details.exectype == Order.ExecType.MARKET
        ]
        stop_orders = [
            order
            for order in orders
            if order.details.exectype
            in [Order.ExecType.STOP, Order.ExecType.STOP_LIMIT]
        ]
        limit_orders = [
            order for order in orders if order.details.exectype == Order.ExecType.LIMIT
        ]

        # Sort stop and limit orders
        stop_orders.sort(key=lambda x: x.details.price, reverse=not is_up_bar)
        limit_orders.sort(key=lambda x: x.details.price, reverse=not is_up_bar)

        # Combine all orders in the specified order
        sorted_orders = market_orders + stop_orders + limit_orders

        return sorted_orders

    def _check_order_fill(
        self, order: Order, bar: Bar
    ) -> Tuple[bool, Optional[ExtendedDecimal], Optional[ExtendedDecimal]]:
        """
        Check if an order should be filled based on the current bar, supporting partial fills.

        Args:
            order (Order): The order to check.
            bar (Bar): The current price bar.
            remaining_size (ExtendedDecimal): The remaining size of the order to be filled.

        Returns:
            Tuple[bool, Optional[ExtendedDecimal], Optional[ExtendedDecimal]]:
                A tuple containing:
                - Whether the order should be filled (bool)
                - The fill price if filled, None otherwise (ExtendedDecimal)
                - The fill size if filled, None otherwise (ExtendedDecimal)
        """
        is_filled, fill_price = order.is_filled(bar)
        return is_filled, fill_price, order.details.size

    def _handle_order_fill(
        self,
        order: Order,
        fill_price: ExtendedDecimal,
        fill_size: ExtendedDecimal,
        timestamp: datetime,
    ) -> None:
        """
        Handle the filling of an order, including partial fills.

        Args:
            order (Order): The order being filled.
            fill_price (ExtendedDecimal): The price at which the order is filled.
            fill_size (ExtendedDecimal): The size of the fill.
            timestamp (datetime): The current timestamp.
        """
        order.on_fill(fill_size, fill_price, timestamp)

        if order.status == Order.Status.FILLED:
            self._handle_order_group_fill(order)
        elif order.status == Order.Status.PARTIALLY_FILLED:
            logger_main.info(
                f"Partial fill for order: {order.id}, remaining size: {order.get_remaining_size()}"
            )

        self.updated_orders.append(order)

    def _handle_order_group_fill(self, filled_order: Order) -> None:
        """
        Handle the implications of a filled order on its order group.

        Args:
            filled_order (Order): The order that was filled.
        """
        if filled_order.order_group:
            if isinstance(filled_order.order_group, OCOGroup):
                self._cancel_other_oco_orders(filled_order)
            elif isinstance(filled_order.order_group, OCAGroup):
                self._cancel_other_oca_orders(filled_order)
            elif isinstance(filled_order.order_group, BracketGroup):
                self._handle_bracket_order_fill(filled_order)

    def _cancel_other_oco_orders(self, filled_order: Order) -> None:
        """
        Cancel other orders in the OCO group when one order is filled.

        Args:
            filled_order (Order): The order that was filled.
        """
        oco_group: OCOGroup = filled_order.order_group
        updated_orders = oco_group.on_order_filled(filled_order)
        self.updated_orders.extend(updated_orders)

    def _cancel_other_oca_orders(self, filled_order: Order) -> None:
        """
        Cancel other orders in the OCA group when one order is filled.

        Args:
            filled_order (Order): The order that was filled.
        """
        oca_group: OCAGroup = filled_order.order_group
        updated_orders = oca_group.on_order_filled(filled_order)
        self.updated_orders.extend(updated_orders)

    def _handle_bracket_order_fill(self, filled_order: Order) -> None:
        """
        Handle the filling of an order in a bracket group.

        Args:
            filled_order (Order): The order that was filled.
        """
        bracket_group: BracketGroup = filled_order.order_group
        updated_orders = bracket_group.on_order_filled(filled_order)
        self.updated_orders.extend(updated_orders)

    # NOT TESTED YET
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

    def cleanup_completed_orders_and_groups(self) -> None:
        # Remove filled, cancelled and rejected order groups
        for order_id, order in self.orders.items():
            if order.status in [
                Order.Status.FILLED,
                Order.Status.CANCELED,
                Order.Status.REJECTED,
            ]:
                del self.orders[order_id]
                logger_main.info(f"Order {order_id} completed and removed.")

        # Remove filled, cancelled and rejected order groups
        for group_id, group in self.order_groups.items():
            if group.get_status() in ["Filled", "Cancelled/Rejected"]:
                del self.order_groups[group_id]
                logger_main.info(f"Order group {group_id} completed and removed.")

        logger_main.info("Cleaned up completed orders and groups")

    def get_updated_orders(self) -> List[Order]:
        return self.updated_orders

    def clear_updated_orders(self) -> None:
        self.updated_orders.clear()

    def modify_order(self, order_id: str, new_details: Dict[str, Any]) -> bool:
        """
        Modify an existing order.

        Args:
            order_id (str): The ID of the order to be modified.
            new_details (Dict[str, Any]): A dictionary containing the new details for the order.

        Returns:
            bool: True if the order was successfully modified, False otherwise.
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        for key, value in new_details.items():
            if hasattr(order.details, key):
                setattr(order.details, key, value)

        self._update_order_group(order)
        self.updated_orders.append(order)
        return True

    def _check_margin_requirements(self, order: Order) -> bool:
        """
        Check if there's sufficient margin to execute the order.

        Args:
            order (Order): The order to check.

        Returns:
            bool: True if there's sufficient margin, False otherwise.
        """
        # Implementation depends on the specific margin requirements of your system
        pass

    def _calculate_fill_size(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> ExtendedDecimal:
        """
        Calculate the fill size for an order based on the order type and current market conditions.

        Args:
            order (Order): The order being executed.
            execution_price (ExtendedDecimal): The price at which the order is being executed.
            bar (Bar): The current price bar.

        Returns:
            ExtendedDecimal: The calculated fill size.
        """
        # Implementation depends on your specific order execution logic
        pass

    def _update_order_group(self, order: Order) -> None:
        """
        Update the order group when an order is modified.

        Args:
            order (Order): The order that was modified.
        """
        if order.order_group:
            order.order_group.update_order(order)

    def _cancel_order(self, order: Order) -> None:
        """
        Cancel an order and update its status.

        Args:
            order (Order): The order to cancel.
        """
        order.cancel()
        self.updated_orders.append(order)
        logger_main.info(f"Cancelled order: {order.id}")

    def _cleanup_completed_orders(self) -> None:
        """
        Remove completed (filled, cancelled, or rejected) orders from pending orders.
        """

        self.orders = {
            id: order
            for id, order in self.orders.items()
            if order.status
            not in [
                Order.Status.FILLED,
                Order.Status.CANCELED,
                Order.Status.REJECTED,
            ]
        }

    # DEPRECATED
    def execute_order(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and handle all aspects of order processing.

        Args:
            order (Order): The order to execute.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing a boolean indicating if the order was executed
            and the resulting Trade object if applicable.

        This method handles:
        - Margin requirement checks
        - Different order types (market, limit, stop)
        - Partial fill logic
        - Trade reversal handling
        """
        if not self._check_margin_requirements(order):
            return False, None

        fill_size = self._calculate_fill_size(order, execution_price, bar)
        if fill_size == ExtendedDecimal("0"):
            return False, None

        trade = self._create_or_update_trade(order, execution_price, fill_size, bar)
        self._update_order_status(order, fill_size)
        self._handle_order_group(order)

        return True, trade

    def _create_or_update_trade(
        self,
        order: Order,
        execution_price: ExtendedDecimal,
        fill_size: ExtendedDecimal,
        bar: Bar,
    ) -> Trade:
        """
        Create a new trade or update an existing one based on the executed order.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            fill_size (ExtendedDecimal): The size of the order fill.
            bar (Bar): The current price bar.

        Returns:
            Trade: The created or updated Trade object.
        """
        # Implementation depends on your trade management logic
        pass

    def _update_order_status(self, order: Order, fill_size: ExtendedDecimal) -> None:
        """
        Update the status of an order after execution.

        Args:
            order (Order): The order to update.
            fill_size (ExtendedDecimal): The size of the order fill.
        """
        if fill_size == order.details.size:
            order.status = Order.Status.FILLED
        elif fill_size > ExtendedDecimal("0"):
            order.status = Order.Status.PARTIALLY_FILLED
        self.updated_orders.append(order)

    def _handle_order_group(self, order: Order) -> None:
        """
        Handle updates to the order group associated with an order.

        Args:
            order (Order): The order that was updated.
        """
        if order.order_group:
            if order.status == Order.Status.FILLED:
                order.order_group.on_order_filled(order)
            elif order.status == Order.Status.CANCELED:
                order.order_group.on_order_cancelled(order)


class TradeManager:
    def __init__(self, commission_rate: ExtendedDecimal):
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.commission_rate = commission_rate
        self.trade_count = 0
        self.updated_trades: List[Trade] = []

    # TESTED
    def manage_trade(
        self,
        order: Order,
        bar: Bar,
        position: "Position",
    ) -> Optional[Trade]:
        """
        Manage trade creation, updates, closures, and reversals.

        Args:
            order (Order): The order being executed.
            bar (Bar): The current price bar.
            position (ExtendedDecimal): The size of the existing position.

        Returns:
            Optional[Trade]: The created or updated Trade object, if applicable.
        """
        fill_size = order.get_filled_size()
        existing_position = position.quantity

        if self._is_trade_reversal(existing_position, order.details.direction):
            return self._handle_trade_reversal(
                order,
                bar,
                existing_position,
            )
        else:
            return self._create_new_trade(order, fill_size, bar)

    def close_trade(
        self,
        trade: Trade,
        order: Order,
        bar: Bar,
    ) -> None:
        """
        Close a specific trade and update the portfolio accordingly.

        Args:
            trade (Trade): The trade to close.
            order (Optional[Order]): The order that triggered the trade closure, if any.
        """

        execution_price = order.get_last_fill_price() or bar.close
        size = order.get_last_fill_size()

        trade.close(order, execution_price, bar, size)

        symbol = trade.ticker
        self.open_trades[symbol].remove(trade)
        if not self.open_trades[symbol]:
            del self.open_trades[symbol]
        self.closed_trades.append(trade)
        self._add_to_updated_trades(trade)

    def partial_close_trade(
        self,
        trade: Trade,
        order: Order,
        bar: Bar,
        close_size: ExtendedDecimal,
    ) -> None:
        """
        Partially close a trade and update its metrics.

        Args:
            trade (Trade): The trade to partially close.
            bar (Bar): The current price bar.
            close_size (ExtendedDecimal): The size of the position to close.
        """
        execution_price = order.get_last_fill_price()

        if close_size > trade.current_size:
            logger_main.warning(
                "Attempted to close more than the current trade size. Closing entire trade."
            )
            close_size = trade.current_size

        trade.close(order, execution_price, bar, close_size)

        if trade.current_size == ExtendedDecimal("0"):
            symbol = trade.ticker
            self.open_trades[symbol].remove(trade)
            if not self.open_trades[symbol]:
                del self.open_trades[symbol]
            self.closed_trades.append(trade)
        else:
            trade.update(bar)

        self._add_to_updated_trades.append(trade)

        logger_main.info(
            f"Partially closed trade {trade.id}. Realized PnL: {trade.get_realized_pnl()}"
        )

    def update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Update all open trades based on current market data.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.
        """
        for symbol, trades in self.open_trades.items():
            if symbol in market_data:
                current_bar = market_data[symbol][min(market_data[symbol].keys())]
                for trade in trades:
                    trade.update(current_bar)

    def clear_updated_trades(self) -> None:
        self.updated_trades.clear()

    def _create_new_trade(
        self,
        order: Order,
        fill_size: ExtendedDecimal,
        bar: Bar,
    ) -> Trade:
        """
        Create a new trade based on an order execution.

        Args:
            order (Order): The order being executed.
            fill_size (ExtendedDecimal): The size of the order fill.
            bar (Bar): The current price bar.

        Returns:
            Trade: The newly created Trade object.
        """
        self.trade_count += 1
        new_trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
            strategy_id=order.details.strategy_id,
        )

        # Set the fill_size
        new_trade.initial_size = fill_size
        new_trade.current_size = fill_size

        symbol = order.details.ticker
        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(new_trade)

        self._add_to_updated_trades(new_trade)
        return new_trade

    def _is_trade_reversal(
        self, existing_position: ExtendedDecimal, order_direction: Order.Direction
    ) -> bool:
        """
        Check if the trade is a reversal.

        Args:
            existing_position (ExtendedDecimal): The existing position size.
            order_direction (Order.Direction): The direction of the new order.

        Returns:
            bool: True if the trade is a reversal, False otherwise.
        """
        return (
            existing_position > ExtendedDecimal("0")
            and order_direction == Order.Direction.SHORT
        ) or (
            existing_position < ExtendedDecimal("0")
            and order_direction == Order.Direction.LONG
        )

    def _handle_trade_reversal(
        self,
        order: Order,
        bar: Bar,
        existing_position: ExtendedDecimal,
    ) -> Trade:
        """
        Handle a trade reversal by closing existing trades and opening a new one.

        Args:
            order (Order): The order causing the reversal.
            bar (Bar): The current price bar.
            existing_position (ExtendedDecimal): The size of the existing position.

        Returns:
            Trade: The new Trade object created after the reversal.
        """

        self._close_existing_trades(order, bar)
        remaining_size = order.get_filled_size() - abs(existing_position)

        if remaining_size > ExtendedDecimal("0"):
            return self._create_new_trade(order, remaining_size, bar)
        return None

    def _close_existing_trades(
        self,
        order: Order,
        bar: Bar,
    ) -> None:
        """
        Close existing trades for a symbol.

        Args:
            order (Order): The order causing the reversal.
            bar (Bar): The current price bar.
        """
        remaining_size = order.get_last_fill_size()
        symbol = order.details.ticker

        # If the order is associated with an order group, first reduce/close the associated trade
        associated_trades: List[Trade] = []
        remaining_trades: List[Trade] = []
        for trade in self.open_trades.get(symbol, []):
            if (
                order.order_group
                and trade.entry_order.order_group
                and trade.entry_order.order_group.id == order.order_group.id
            ):
                associated_trades.append(trade)
            else:
                remaining_trades.append(trade)

        associated_trades.extend(remaining_trades)

        for trade in associated_trades[:]:
            if remaining_size <= ExtendedDecimal("0"):
                break

            if remaining_size >= trade.current_size:
                self.close_trade(trade, order, bar)
                remaining_size -= trade.current_size
            else:
                self.partial_close_trade(trade, order, bar, remaining_size)
                remaining_size = ExtendedDecimal("0")

    def _add_to_updated_trades(self, trade: Trade) -> None:
        if trade not in self.updated_trades:
            self.updated_trades.append(trade)

    # NOT CONFIRMED
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
            trade.get_unrealized_pnl() for trade in self.open_trades.get(symbol, [])
        )

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        return self.get_open_trades(strategy_id) + self.get_closed_trades(strategy_id)

    def _get_trade_by_order(self, order: Order) -> Optional[Trade]:
        """
        Get the trade associated with a given order.

        Args:
            order (Order): The order to look up.

        Returns:
            Optional[Trade]: The associated Trade object, if found.
        """
        for trades in self.open_trades.values():
            for trade in trades:
                if trade.entry_order == order:
                    return trade
        return None

    def _update_average_entry_price(
        self, trade: Trade, new_size: ExtendedDecimal, new_price: ExtendedDecimal
    ) -> None:
        """
        Update the average entry price of a trade when adding to the position.

        Args:
            trade (Trade): The trade to update.
            new_size (ExtendedDecimal): The size of the new addition to the position.
            new_price (ExtendedDecimal): The price of the new addition.
        """
        total_size = trade.current_size + new_size
        trade.entry_price = (
            (trade.entry_price * trade.current_size) + (new_price * new_size)
        ) / total_size

    def _get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """
        Get the most recent open trade for a symbol.

        Args:
            symbol (str): The symbol to get the latest trade for.

        Returns:
            Optional[Trade]: The most recent open trade for the symbol, if any.
        """
        trades = self.open_trades.get(symbol, [])
        return trades[-1] if trades else None

    def _get_position_size(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current position size for a symbol.

        Args:
            symbol (str): The symbol to get the position size for.

        Returns:
            ExtendedDecimal: The current position size (positive for long, negative for short).
        """
        return sum(
            trade.current_size * trade.direction.value
            for trade in self.open_trades.get(symbol, [])
        )

    def get_average_entry_price(self, symbol: str) -> Optional[ExtendedDecimal]:
        """
        Get the average entry price for a symbol across all open trades.

        Args:
            symbol (str): The symbol to get the average entry price for.

        Returns:
            Optional[ExtendedDecimal]: The average entry price, or None if there are no open trades.
        """
        trades = self.open_trades.get(symbol, [])
        if not trades:
            return None

        total_size = sum(abs(trade.current_size) for trade in trades)
        weighted_price_sum = sum(
            trade.entry_price * abs(trade.current_size) for trade in trades
        )
        return weighted_price_sum / total_size if total_size > 0 else None


class AccountManager:
    def __init__(self, initial_capital: ExtendedDecimal, margin_ratio: ExtendedDecimal):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.margin_ratio = margin_ratio
        self.margin_used = ExtendedDecimal("0")
        self.equity = initial_capital
        self.unrealized_pnl = ExtendedDecimal("0")
        self.realized_pnl = ExtendedDecimal("0")
        self.transaction_log: List[Dict] = []

    @property
    def buying_power(self) -> None:
        """
        Calculate and return the current buying power.

        Returns:
            ExtendedDecimal: The current buying power.
        """
        return self.equity / self.margin_ratio - self.margin_used

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

    def update_margin(self, margin_change: ExtendedDecimal) -> None:
        """
        Update the margin used.

        Args:
            margin_change (ExtendedDecimal): The change in margin used.
        """
        self.margin_used += margin_change

    def update_cash(self, amount: ExtendedDecimal, reason: str) -> None:
        """
        Update the cash balance.

        Args:
            amount (ExtendedDecimal): The amount to add (positive) or subtract (negative) from the cash balance.
            reason (str): The reason for the cash update.
        """
        self.cash += amount
        logger_main.info(
            f"Cash updated: {amount} due to {reason}. New balance: {self.cash}"
        )

    def update_equity(self, unrealized_pnl: ExtendedDecimal) -> None:
        """
        Update the account equity based on unrealized PnL.

        Args:
            unrealized_pnl (ExtendedDecimal): The current unrealized PnL.
        """
        self.unrealized_pnl = unrealized_pnl
        self.equity = self.cash + self.unrealized_pnl
        logger_main.info(f"Equity updated. New equity: {self.equity}")

    def update_margin_used(
        self, long_value: ExtendedDecimal, short_value: ExtendedDecimal
    ) -> None:
        """
        Update the margin used based on current long and short position values.

        Args:
            long_value (ExtendedDecimal): The total value of long positions.
            short_value (ExtendedDecimal): The total value of short positions.
        """
        long_margin = long_value * self.margin_ratio
        short_margin = (
            short_value * self.margin_ratio * ExtendedDecimal("1.5")
        )  # Higher margin for short positions
        self.margin_used = long_margin + short_margin
        logger_main.info(f"Margin used updated. New margin used: {self.margin_used}")

    def realize_pnl(self, amount: ExtendedDecimal) -> None:
        """
        Realize PnL and update cash balance.

        Args:
            amount (ExtendedDecimal): The amount of PnL to realize.
        """
        self.realized_pnl += amount
        self.cash += amount
        self.equity += amount
        logger_main.info(
            f"Realized PnL: {amount}. New realized PnL: {self.realized_pnl}"
        )


class Position:
    def __init__(self, symbol: str):
        self.symbol: str = symbol
        self.quantity: ExtendedDecimal = ExtendedDecimal("0")
        self.average_price: ExtendedDecimal = ExtendedDecimal("0")
        self.total_cost: ExtendedDecimal = ExtendedDecimal("0")

    def update_position(
        self, quantity: ExtendedDecimal, price: ExtendedDecimal
    ) -> None:
        new_quantity = self.quantity + quantity
        new_cost = self.total_cost + (quantity * price)

        if new_quantity != ExtendedDecimal("0"):
            self.average_price = new_cost / new_quantity
        else:
            self.average_price = ExtendedDecimal("0")

        self.quantity = new_quantity
        self.total_cost = new_cost

    def calculate_unrealized_pnl(
        self, current_price: ExtendedDecimal
    ) -> ExtendedDecimal:
        return (current_price - self.average_price) * self.quantity


class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def get_all_positions(self) -> Dict[str, Position]:
        return self.positions

    def calculate_position_value(
        self, symbol: str, current_price: ExtendedDecimal
    ) -> ExtendedDecimal:
        """
        Calculate the current value of a position.

        Args:
            symbol (str): The symbol of the position.
            current_price (ExtendedDecimal): The current market price of the symbol.

        Returns:
            ExtendedDecimal: The current value of the position.
        """
        position = self.get_position(symbol)
        return position.quantity * current_price

    def update_position(
        self, order: Order, execution_price: ExtendedDecimal, fill_size: ExtendedDecimal
    ) -> None:
        """
        Update the position for a given symbol based on an executed order.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            fill_size (ExtendedDecimal): The size of the order fill.
        """
        symbol = order.details.ticker
        position = self.get_position(symbol)
        direction = order.details.direction.value
        quantity = fill_size * ExtendedDecimal(str(direction))
        position.update_position(quantity, execution_price)

        logger_main.info(f"Updated position for {symbol}: {position}")

    def get_long_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        """
        Calculate the total value of all long positions.

        Args:
            current_prices (Dict[str, ExtendedDecimal]): Current prices for all symbols.

        Returns:
            ExtendedDecimal: The total value of all long positions.
        """
        return sum(
            position.quantity * current_prices[symbol]
            for symbol, position in self.positions.items()
            if position.quantity > 0
        )

    def get_short_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        """
        Calculate the total value of all short positions.

        Args:
            current_prices (Dict[str, ExtendedDecimal]): Current prices for all symbols.

        Returns:
            ExtendedDecimal: The total value of all short positions.
        """
        return sum(
            abs(position.quantity) * current_prices[symbol]
            for symbol, position in self.positions.items()
            if position.quantity < 0
        )

    def get_position(self, symbol: str) -> Position:
        """
        Get the current position for a symbol.

        Args:
            symbol (str): The symbol to get the position for.

        Returns:
            ExtendedDecimal: The current position size (positive for long, negative for short).
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def get_total_position_value(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        """
        Calculate the total value of all positions.

        Args:
            current_prices (Dict[str, ExtendedDecimal]): A dictionary of current prices for all symbols.

        Returns:
            ExtendedDecimal: The total value of all positions.
        """
        return sum(
            self.calculate_position_value(symbol, current_prices[symbol])
            for symbol in self.positions
        )

    def get_total_unrealized_pnl(
        self, current_prices: Dict[str, ExtendedDecimal]
    ) -> ExtendedDecimal:
        """
        Get the total unrealized PnL across all positions.

        Returns:
            ExtendedDecimal: The total unrealized PnL.
        """
        return sum(
            position.calculate_unrealized_pnl(current_prices[position.symbol])
            for position in self.positions.values()
        )

    def get_total_realized_pnl(self) -> ExtendedDecimal:
        return sum(position.realized_pnl for position in self.positions.values())

    def generate_close_orders(
        self, timestamp: datetime, timeframe: Timeframe
    ) -> List[Order]:
        """
        Generate orders to close all current positions.

        Args:
            timestamp (datetime): The current timestamp.
            timeframe (Timeframe): The timeframe for the close orders.

        Returns:
            List[Order]: A list of orders to close all positions.
        """
        close_orders = []
        for symbol, position in self.positions.items():
            order_details = OrderDetails(
                ticker=symbol,
                direction=Order.Direction.SHORT
                if position > 0
                else Order.Direction.LONG,
                size=abs(position),
                price=None,  # Market order
                exectype=Order.ExecType.MARKET,
                timestamp=timestamp,
                timeframe=timeframe,
                strategy_id="CLOSE_ALL",
            )
            close_orders.append(Order(str(uuid.uuid4()), order_details))
        return close_orders

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
        self.initial_capital = initial_capital  # Removed in latest version
        self.max_risk = max_risk
        self.max_risk_per_trade = max_risk_per_trade
        self.max_risk_per_symbol = max_risk_per_symbol  # Removed in latest version
        self.max_drawdown = max_drawdown
        self.var_confidence_level = var_confidence_level
        self.margin_ratio = margin_ratio  # Removed in latest version
        self.margin_call_threshold = margin_call_threshold  # Removed in latest version
        self.equity_history: List[ExtendedDecimal] = [
            initial_capital
        ]  # Removed in latest version
        self.symbol_weights: Dict[
            str, ExtendedDecimal
        ] = {}  # Removed in latest version
        self._initialize_symbol_weights(symbols)  # Removed in latest version

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

    def check_drawdown(self, equity: ExtendedDecimal) -> bool:
        return self.calculate_drawdown(equity) <= self.max_drawdown

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

    def check_margin_requirements(
        self,
        order: Order,
        account_value: ExtendedDecimal,
        current_positions: Dict[str, Position],
    ) -> bool:
        """
        Perform comprehensive margin and risk checks for an order.

        This method checks:
        1. Margin requirements
        2. Position size limits
        3. Per-symbol risk limits
        4. Overall portfolio risk limits

        Args:
            order (Order): The order to check.
            account_value (ExtendedDecimal): The current account value.
            current_positions (Dict[str, ExtendedDecimal]): The current positions.

        Returns:
            bool: True if the order passes all checks, False otherwise.
        """
        symbol = order.details.ticker
        fill_price = order.details.price

        current_position = current_positions.get(symbol, None)
        if current_position:
            current_position_size = current_position.quantity
        else:
            current_position_size = ExtendedDecimal("0")

        new_position_size = current_position_size + order.details.size

        # Check margin requirement
        required_margin = order.details.size * fill_price * self.margin_ratio
        if required_margin > account_value:
            logger_main.warning(
                f"Insufficient margin. Required: {required_margin}, Available: {account_value}"
            )
            return False

        # Check position size limits
        max_position_size = account_value * self.max_risk_per_symbol
        if abs(new_position_size) > max_position_size:
            logger_main.warning(
                f"Position size ({new_position_size}) exceeds maximum allowed ({max_position_size})"
            )
            return False

        # Check per-symbol risk limit
        symbol_exposure = abs(new_position_size * fill_price)
        max_symbol_risk = account_value * self.max_risk_per_symbol
        if symbol_exposure > max_symbol_risk:
            logger_main.warning(
                f"Symbol exposure ({symbol_exposure}) exceeds maximum allowed ({max_symbol_risk})"
            )
            return False

        # Check overall portfolio risk limit
        new_total_exposure = sum(
            (pos.quantity + (order.details.size if sym == symbol else 0)) * fill_price
            for sym, pos in current_positions.items()
        )
        max_portfolio_risk = account_value * self.max_risk
        if new_total_exposure > max_portfolio_risk:
            logger_main.warning(
                f"Portfolio exposure ({new_total_exposure}) would exceed maximum allowed ({max_portfolio_risk})"
            )
            return False

        return True

    def calculate_position_size(
        self,
        symbol: str,
        account_value: ExtendedDecimal,
        risk_per_trade: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        stop_loss: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Calculate the position size based on risk parameters.

        Args:
            symbol (str): The symbol for the trade.
            account_value (ExtendedDecimal): The current account value.
            risk_per_trade (ExtendedDecimal): The risk percentage per trade.
            entry_price (ExtendedDecimal): The entry price for the trade.
            stop_loss (ExtendedDecimal): The stop loss price for the trade.

        Returns:
            ExtendedDecimal: The calculated position size.
        """
        risk_amount = (
            account_value
            * risk_per_trade
            * self.symbol_weights.get(symbol, ExtendedDecimal("1"))
        )
        price_difference = abs(entry_price - stop_loss)
        position_size = risk_amount / price_difference

        # Ensure position size doesn't exceed max_position_size
        return min(position_size, self.max_position_size)

    def calculate_drawdown(
        self, current_equity: ExtendedDecimal, peak_equity: ExtendedDecimal
    ) -> ExtendedDecimal:
        """
        Calculate the current drawdown.

        Args:
            current_equity (ExtendedDecimal): The current equity.
            peak_equity (ExtendedDecimal): The peak equity reached.

        Returns:
            ExtendedDecimal: The current drawdown as a percentage.
        """
        return (peak_equity - current_equity) / peak_equity

    def _calculate_margin_usage(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
    ) -> ExtendedDecimal:
        """
        Calculate the total margin usage for given positions.

        Args:
            positions (Dict[str, ExtendedDecimal]): The current positions.
            current_prices (Dict[str, ExtendedDecimal]): The current market prices.

        Returns:
            ExtendedDecimal: The total margin usage.
        """
        return sum(
            abs(size) * current_prices[symbol] * self.margin_ratio
            for symbol, size in positions.items()
        )

    def check_margin_call(
        self,
        equity: ExtendedDecimal,
        margin_used: ExtendedDecimal,
        margin_call_threshold: ExtendedDecimal,
    ) -> bool:
        """
        Check if a margin call should be triggered.

        Args:
            equity (ExtendedDecimal): Current account equity.
            margin_used (ExtendedDecimal): Current margin used.
            margin_call_threshold (ExtendedDecimal): The threshold for triggering a margin call.

        Returns:
            bool: True if a margin call should be triggered, False otherwise.
        """
        if margin_used > ExtendedDecimal("0"):
            return equity / margin_used < margin_call_threshold
        return False

    def calculate_var(self, returns: np.ndarray) -> ExtendedDecimal:
        """
        Calculate Value at Risk (VaR) using the historical method.

        Args:
            returns (np.ndarray): Array of historical returns.

        Returns:
            ExtendedDecimal: The calculated VaR.
        """
        var = np.percentile(returns, (1 - self.var_confidence_level) * 100)
        return ExtendedDecimal(str(var))

    def calculate_cvar(self, returns: np.ndarray) -> ExtendedDecimal:
        """
        Calculate Conditional Value at Risk (CVaR) using the historical method.

        Args:
            returns (np.ndarray): Array of historical returns.

        Returns:
            ExtendedDecimal: The calculated CVaR.
        """
        var = self.calculate_var(returns)
        cvar = np.mean(returns[returns <= float(var)])
        return ExtendedDecimal(str(cvar))

    def check_risk_limits(
        self,
        order: Order,
        fill_size: ExtendedDecimal,
        account_value: ExtendedDecimal,
        current_positions: Dict[str, Position],
    ) -> bool:
        """
        Check if an order's risk is within the set limits.

        Args:
            order (Order): The order being checked.
            fill_size (ExtendedDecimal): The size of the order fill.
            account_value (ExtendedDecimal): The current account value.
            current_positions (Dict[str, Position]): The current positions.

        Returns:
            bool: True if the order is within risk limits, False otherwise.
        """
        symbol = order.details.ticker
        fill_price = order.get_last_fill_price()
        order_value = fill_size * fill_price

        # Check max risk per trade
        if order_value > account_value * self.max_risk_per_trade:
            logger_main.warning(
                f"Order value {order_value} exceeds max risk per trade {self.max_risk_per_trade * account_value}"
            )
            return False

        # Check max risk per symbol
        current_position = current_positions.get(symbol, Position(symbol))
        new_position_value = (current_position.quantity + fill_size) * fill_price
        if new_position_value > account_value * self.max_risk_per_symbol:
            logger_main.warning(
                f"New position value {new_position_value} for {symbol} exceeds max risk per symbol {self.max_risk_per_symbol * account_value}"
            )
            return False

        # Check total portfolio risk
        total_position_value = sum(
            (pos.quantity + (fill_size if sym == symbol else ExtendedDecimal("0")))
            * fill_price
            for sym, pos in current_positions.items()
        )
        if total_position_value > account_value * self.max_risk:
            logger_main.warning(
                f"Total position value {total_position_value} would exceed max portfolio risk {self.max_risk * account_value}"
            )
            return False

        return True

    def calculate_unrealized_pnl(
        self,
        position: ExtendedDecimal,
        entry_price: ExtendedDecimal,
        current_price: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Calculate the unrealized P&L for a position.

        Args:
            position (ExtendedDecimal): The position size.
            entry_price (ExtendedDecimal): The average entry price of the position.
            current_price (ExtendedDecimal): The current market price.

        Returns:
            ExtendedDecimal: The unrealized P&L.
        """
        return (current_price - entry_price) * position

    def handle_margin_call(
        self,
        positions: Dict[str, ExtendedDecimal],
        current_prices: Dict[str, ExtendedDecimal],
        open_trades: List[Trade],
        equity: ExtendedDecimal,
        margin_used: ExtendedDecimal,
        margin_call_threshold: ExtendedDecimal,
    ) -> List[Tuple[str, ExtendedDecimal]]:
        """
        Handle a margin call by gradually reducing positions.

        Args:
            positions (Dict[str, ExtendedDecimal]): Current positions keyed by symbol.
            current_prices (Dict[str, ExtendedDecimal]): Current market prices keyed by symbol.
            open_trades (List[Trade]): List of all open trades.
            equity (ExtendedDecimal): Current account equity.
            margin_used (ExtendedDecimal): Current margin used.
            margin_call_threshold (ExtendedDecimal): The threshold for triggering a margin call.

        Returns:
            List[Tuple[str, ExtendedDecimal]]: A list of tuples containing the symbol and amount to reduce for each position.
        """
        positions_to_reduce = []
        while self.check_margin_call(equity, margin_used, margin_call_threshold):
            # Calculate unrealized P&L for each position
            unrealized_pnls = {
                symbol: self.calculate_unrealized_pnl(
                    pos, trade.entry_price, current_prices[symbol]
                )
                for symbol, pos in positions.items()
                for trade in open_trades
                if trade.ticker == symbol
            }

            # Sort positions by unrealized P&L (most negative first)
            sorted_positions = sorted(unrealized_pnls.items(), key=lambda x: x[1])

            if not sorted_positions:
                logger_main.error(
                    "Unable to resolve margin call. No suitable positions to reduce."
                )
                break

            symbol, _ = sorted_positions[0]
            reduce_amount = min(
                abs(positions[symbol]), abs(positions[symbol]) * ExtendedDecimal("0.1")
            )  # Reduce by 10% or full position

            positions_to_reduce.append((symbol, reduce_amount))

            # Update positions and recalculate margin
            positions[symbol] -= (
                reduce_amount if positions[symbol] > 0 else -reduce_amount
            )
            position_value = sum(
                abs(pos) * current_prices[sym] for sym, pos in positions.items()
            )
            margin_used = position_value * margin_call_threshold
            equity = equity - (
                reduce_amount * current_prices[symbol]
            )  # Assuming worst-case scenario

            logger_main.info(
                f"Reducing position in {symbol} by {reduce_amount} to address margin call."
            )

        return positions_to_reduce

    def calculate_portfolio_var(
        self, returns: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> ExtendedDecimal:
        """
        Calculate portfolio Value at Risk (VaR) using the variance-covariance method.

        Args:
            returns (Dict[str, np.ndarray]): Dictionary of historical returns for each asset.
            weights (Dict[str, float]): Dictionary of portfolio weights for each asset.

        Returns:
            ExtendedDecimal: The calculated portfolio VaR.
        """
        # Combine returns into a single matrix
        return_matrix = np.column_stack([returns[asset] for asset in weights.keys()])

        # Calculate the covariance matrix
        cov_matrix = np.cov(return_matrix.T)

        # Calculate portfolio variance
        portfolio_variance = np.dot(
            np.dot(list(weights.values()), cov_matrix), list(weights.values())
        )

        # Calculate portfolio VaR
        z_score = stats.norm.ppf(1 - self.var_confidence_level)
        portfolio_var = z_score * np.sqrt(portfolio_variance)

        return ExtendedDecimal(str(portfolio_var))

    def calculate_portfolio_cvar(
        self, returns: Dict[str, np.ndarray], weights: Dict[str, float]
    ) -> ExtendedDecimal:
        """
        Calculate portfolio Conditional Value at Risk (CVaR) using the historical method.

        Args:
            returns (Dict[str, np.ndarray]): Dictionary of historical returns for each asset.
            weights (Dict[str, float]): Dictionary of portfolio weights for each asset.

        Returns:
            ExtendedDecimal: The calculated portfolio CVaR.
        """
        # Combine returns into a single array, weighted by portfolio weights
        portfolio_returns = sum(
            returns[asset] * weights[asset] for asset in weights.keys()
        )

        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - self.var_confidence_level) * 100)

        # Calculate CVaR
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])

        return ExtendedDecimal(str(cvar))
