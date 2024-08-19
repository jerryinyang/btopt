import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .data.bar import Bar
from .data.timeframe import Timeframe
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


class OrderGroupType(Enum):
    OCO = "One-Cancels-the-Other"
    OCA = "One-Cancels-All"
    BRACKET = "Bracket"


class OrderGroup(ABC):
    """
    Abstract base class for order groups.
    """

    def __init__(self, group_type: OrderGroupType):
        self.id: str = str(uuid.uuid4())
        self.type: OrderGroupType = group_type
        self.orders: List[Order] = []

    @abstractmethod
    def add_order(self, order: "Order") -> None:
        """Add an order to the group."""
        pass

    @abstractmethod
    def remove_order(self, order: "Order") -> None:
        """Remove an order from the group."""
        pass

    @abstractmethod
    def on_order_filled(self, filled_order: "Order") -> None:
        """Handle the event when an order in the group is filled."""
        pass

    @abstractmethod
    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        """Handle the event when an order in the group is cancelled."""
        pass

    @abstractmethod
    def on_order_rejected(self, rejected_order: "Order") -> None:
        """Handle the event when an order in the group is rejected."""
        pass

    def get_status(self) -> str:
        """Get the overall status of the order group."""
        if not self.orders:
            return "Empty"
        if all(order.status == Order.Status.FILLED for order in self.orders):
            return "Filled"
        if all(
            order.status in [Order.Status.CANCELED, Order.Status.REJECTED]
            for order in self.orders
        ):
            return "Cancelled/Rejected"
        return "Active"


class OCOGroup(OrderGroup):
    """
    Represents a One-Cancels-the-Other order group.
    """

    def __init__(self):
        super().__init__(OrderGroupType.OCO)

    def add_order(self, order: "Order") -> None:
        if len(self.orders) < 2:
            self.orders.append(order)
            order.order_group = self
        else:
            logger_main.warning("OCO group can only contain two orders.")

    def remove_order(self, order: "Order") -> None:
        if order in self.orders:
            self.orders.remove(order)
            order.order_group = None

    def on_order_filled(self, filled_order: "Order") -> None:
        for order in self.orders:
            if order != filled_order and order.status != Order.Status.FILLED:
                order.cancel()

    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        # In OCO, cancelling one order doesn't affect the other
        pass

    def on_order_rejected(self, rejected_order: "Order") -> None:
        # In OCO, rejecting one order doesn't affect the other
        pass


class OCAGroup(OrderGroup):
    """
    Represents a One-Cancels-All order group.
    """

    def __init__(self):
        super().__init__(OrderGroupType.OCA)

    def add_order(self, order: "Order") -> None:
        self.orders.append(order)
        order.order_group = self

    def remove_order(self, order: "Order") -> None:
        if order in self.orders:
            self.orders.remove(order)
            order.order_group = None

    def on_order_filled(self, filled_order: "Order") -> None:
        for order in self.orders:
            if order != filled_order and order.status != Order.Status.FILLED:
                order.cancel()

    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        # In OCA, cancelling one order cancels all others
        for order in self.orders:
            if order != cancelled_order and order.status != Order.Status.CANCELED:
                order.cancel()

    def on_order_rejected(self, rejected_order: "Order") -> None:
        # In OCA, rejecting one order doesn't affect the others
        pass


class BracketGroup(OrderGroup):
    """
    Represents a Bracket order group.
    """

    def __init__(self):
        super().__init__(OrderGroupType.BRACKET)
        self.entry_order: Optional[Order] = None
        self.take_profit_order: Optional[Order] = None
        self.stop_loss_order: Optional[Order] = None

    def add_order(self, order: "Order") -> None:
        if not self.entry_order:
            self.entry_order = order
        elif not self.take_profit_order:
            self.take_profit_order = order
        elif not self.stop_loss_order:
            self.stop_loss_order = order
        else:
            logger_main.warning("Bracket group already has all required orders.")

        self.orders.append(order)
        order.order_group = self

    def remove_order(self, order: "Order") -> None:
        if order in self.orders:
            self.orders.remove(order)
            order.order_group = None
            if order == self.entry_order:
                self.entry_order = None
            elif order == self.take_profit_order:
                self.take_profit_order = None
            elif order == self.stop_loss_order:
                self.stop_loss_order = None

    def on_order_filled(self, filled_order: "Order") -> None:
        if filled_order == self.entry_order:
            # Activate take-profit and stop-loss orders
            if self.take_profit_order:
                self.take_profit_order.activate()
            if self.stop_loss_order:
                self.stop_loss_order.activate()
        elif filled_order in [self.take_profit_order, self.stop_loss_order]:
            # Cancel the other exit order
            other_exit_order = (
                self.take_profit_order
                if filled_order == self.stop_loss_order
                else self.stop_loss_order
            )
            if other_exit_order and other_exit_order.status != Order.Status.FILLED:
                other_exit_order.cancel()

    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        if cancelled_order == self.entry_order:
            # Cancel both exit orders
            for order in [self.take_profit_order, self.stop_loss_order]:
                if order and order.status != Order.Status.CANCELED:
                    order.cancel()

    def on_order_rejected(self, rejected_order: "Order") -> None:
        if rejected_order == self.entry_order:
            # Cancel both exit orders
            for order in [self.take_profit_order, self.stop_loss_order]:
                if order and order.status != Order.Status.CANCELED:
                    order.cancel()


@dataclass
class Fill:
    """Represents a single fill for an order."""

    price: ExtendedDecimal
    size: ExtendedDecimal
    timestamp: datetime


@dataclass(frozen=True)
class OrderDetails:
    """A frozen dataclass representing the details of an order."""

    ticker: str
    direction: "Order.Direction"
    size: ExtendedDecimal
    price: ExtendedDecimal
    exectype: "Order.ExecType"
    timestamp: datetime
    timeframe: Timeframe
    strategy_id: Optional[str]
    expiry: Optional[datetime] = None
    parent_id: Optional[int] = None
    stoplimit_price: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    exit_profit: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    exit_loss: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    exit_profit_percent: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    exit_loss_percent: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    trailing_percent: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    slippage: Optional[ExtendedDecimal] = field(default_factory=lambda: None)


class Order:
    """Represents a trading order with various execution types and statuses."""

    class Direction(Enum):
        """Enum representing the direction of an order."""

        LONG = 1
        SHORT = -1

    class ExecType(Enum):
        """Enum representing the execution type of an order."""

        MARKET = "Market"
        LIMIT = "Limit"
        STOP = "Stop"
        STOP_LIMIT = "StopLimit"
        EXIT_LIMIT = "ExitLimit"
        EXIT_STOP = "ExitStop"
        TRAILING = "Trailing"

    class Status(Enum):
        """Enum representing the current status of an order."""

        CREATED = "Created"
        ACCEPTED = "Accepted"
        PARTIALLY_FILLED = "Partially Filled"
        FILLED = "Filled"
        CANCELED = "Canceled"
        REJECTED = "Rejected"

    def __init__(self, order_id: str, details: OrderDetails):
        """
        Initialize a new Order instance.

        Args:
            order_id (str): A unique identifier for the order.
            details (OrderDetails): The details of the order.
        """
        self.id: str = order_id
        self.details: OrderDetails = details
        self.status: Order.Status = self.Status.CREATED
        self.fills: List[Fill] = []
        self.order_group: Optional[OrderGroup] = None
        self.is_active: bool = True

    def is_filled(self, bar: Bar) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check if the order is filled based on the current price bar."""
        if self.status == self.Status.FILLED:
            return True, self.get_last_fill_price()

        filled, price = self._check_fill_conditions(bar)
        if filled:
            self.on_fill(self.get_remaining_size(), price, bar.timestamp)
        return filled, price

    def _check_fill_conditions(
        self, bar: Bar
    ) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check if the order's fill conditions are met based on the current price bar."""
        if self.details.exectype == self.ExecType.MARKET:
            return True, self._apply_slippage(bar.open)

        if self.details.direction == self.Direction.LONG:
            return self._check_long_fill_conditions(bar)
        else:
            return self._check_short_fill_conditions(bar)

    def _check_long_fill_conditions(
        self, bar: "Bar"
    ) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check fill conditions for long orders."""
        if self.details.exectype in [self.ExecType.LIMIT, self.ExecType.EXIT_LIMIT]:
            if bar.open <= self.details.price:
                return True, self._apply_slippage(bar.open)
            elif bar.low <= self.details.price <= bar.high:
                return True, self._apply_slippage(self.details.price)

        elif self.details.exectype in [self.ExecType.STOP, self.ExecType.EXIT_STOP]:
            if bar.open >= self.details.price:
                return True, self._apply_slippage(bar.open)
            elif bar.low <= self.details.price <= bar.high:
                return True, self._apply_slippage(self.details.price)

        elif self.details.exectype == self.ExecType.TRAILING:
            return self._check_trailing_stop(bar, is_long=True)

        elif self.details.exectype == self.ExecType.STOP_LIMIT:
            return self._check_stop_limit(bar, is_long=True)

        return False, None

    def _check_short_fill_conditions(
        self, bar: "Bar"
    ) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check fill conditions for short orders."""
        if self.details.exectype in [self.ExecType.LIMIT, self.ExecType.EXIT_LIMIT]:
            if bar.open >= self.details.price:
                return True, self._apply_slippage(bar.open)
            elif bar.low <= self.details.price <= bar.high:
                return True, self._apply_slippage(self.details.price)

        elif self.details.exectype in [self.ExecType.STOP, self.ExecType.EXIT_STOP]:
            if bar.open <= self.details.price:
                return True, self._apply_slippage(bar.open)
            elif bar.low <= self.details.price <= bar.high:
                return True, self._apply_slippage(self.details.price)

        elif self.details.exectype == self.ExecType.TRAILING:
            return self._check_trailing_stop(bar, is_long=False)

        elif self.details.exectype == self.ExecType.STOP_LIMIT:
            return self._check_stop_limit(bar, is_long=False)

        return False, None

    def _check_trailing_stop(
        self, bar: "Bar", is_long: bool
    ) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check and update trailing stop conditions."""
        if self.trailing_activation_price is None:
            self.trailing_activation_price = bar.high if is_long else bar.low
            self.details.price = self._calculate_trailing_stop_price(
                self.trailing_activation_price, is_long
            )
        else:
            if (is_long and bar.high > self.trailing_activation_price) or (
                not is_long and bar.low < self.trailing_activation_price
            ):
                self.trailing_activation_price = bar.high if is_long else bar.low
                self.details.price = self._calculate_trailing_stop_price(
                    self.trailing_activation_price, is_long
                )

        if (is_long and bar.low <= self.details.price) or (
            not is_long and bar.high >= self.details.price
        ):
            return True, self._apply_slippage(self.details.price)

        return False, None

    def _calculate_trailing_stop_price(
        self, activation_price: ExtendedDecimal, is_long: bool
    ) -> ExtendedDecimal:
        """Calculate the trailing stop price based on the activation price and trailing percentage."""
        if self.details.trailing_percent is None:
            logger_main.log_and_raise(
                ValueError("Trailing percent is not set for trailing stop order")
            )

        trailing_amount = activation_price * self.details.trailing_percent
        return (
            activation_price - trailing_amount
            if is_long
            else activation_price + trailing_amount
        )

    def _check_stop_limit(
        self, bar: "Bar", is_long: bool
    ) -> tuple[bool, Optional[ExtendedDecimal]]:
        """Check fill conditions for stop-limit orders."""
        if not hasattr(self, "stop_triggered"):
            self.stop_triggered = False

        if not self.stop_triggered:
            if (
                is_long
                and (
                    bar.open >= self.details.stoplimit_price
                    or bar.low <= self.details.stoplimit_price <= bar.high
                )
            ) or (
                not is_long
                and (
                    bar.open <= self.details.stoplimit_price
                    or bar.low <= self.details.stoplimit_price <= bar.high
                )
            ):
                self.stop_triggered = True
        else:
            return (
                self._check_long_fill_conditions(bar)
                if is_long
                else self._check_short_fill_conditions(bar)
            )

        return False, None

    def on_fill(
        self,
        fill_size: ExtendedDecimal,
        fill_price: ExtendedDecimal,
        timestamp: datetime,
    ) -> None:
        """
        Handle the event when the order is filled (partially or fully).

        Args:
            fill_size (ExtendedDecimal): The size of the fill.
            fill_price (ExtendedDecimal): The price of the fill.
            timestamp (datetime): The timestamp of the fill.
        """
        self.fills.append(Fill(fill_price, fill_size, timestamp))
        filled_size = sum(fill.size for fill in self.fills)

        if filled_size < self.details.size:
            self.status = self.Status.PARTIALLY_FILLED
        else:
            self.status = self.Status.FILLED
            self.is_active = False

        if self.order_group:
            self.order_group.on_order_filled(self)

        logger_main.info(
            f"Order {self.id} filled: size {fill_size}, price {fill_price}"
        )

    def on_cancel(self) -> None:
        """Handle the event when the order is cancelled."""
        self.status = self.Status.CANCELED
        self.is_active = False
        if self.order_group:
            self.order_group.on_order_cancelled(self)
        logger_main.info(f"Order {self.id} cancelled")

    def on_reject(self, reason: str) -> None:
        """
        Handle the event when the order is rejected.

        Args:
            reason (str): The reason for the rejection.
        """
        self.status = self.Status.REJECTED
        self.is_active = False
        if self.order_group:
            self.order_group.on_order_rejected(self)
        logger_main.info(f"Order {self.id} rejected: {reason}")

    def cancel(self) -> None:
        """Request cancellation of the order."""
        if self.is_active:
            self.on_cancel()

    def activate(self) -> None:
        """Activate the order (used for stop and limit orders in bracket groups)."""
        if self.status == self.Status.CREATED:
            self.status = self.Status.ACCEPTED
            logger_main.info(f"Order {self.id} activated")

    def __repr__(self) -> str:
        """Return a string representation of the Order."""
        return (
            f"Order(id={self.id}, ticker={self.details.ticker}, "
            f"direction={self.details.direction.name}, "
            f"price={self.details.price}, status={self.status.name}, "
            f"exectype={self.details.exectype.name}, "
            f"filled={self.get_filled_size()}/{self.details.size})"
        )

    def get_filled_size(self) -> ExtendedDecimal:
        """Get the total filled size of the order."""
        return sum(fill.size for fill in self.fills)

    def is_expired(self, current_time: datetime) -> bool:
        """Check if the order has expired."""
        return self.details.expiry is not None and current_time >= self.details.expiry

    def update_size(self, new_size: ExtendedDecimal):
        """Update the order size, ensuring it doesn't decrease below the filled size."""
        if new_size < self.get_filled_size():
            logger_main.log_and_raise(
                ValueError("New size cannot be less than the filled size")
            )
        self.details.size = new_size

    def get_remaining_size(self) -> ExtendedDecimal:
        """Get the remaining unfilled size of the order."""
        return self.details.size - self.get_filled_size()

    def get_average_fill_price(self) -> Optional[ExtendedDecimal]:
        """Get the average fill price of the order."""
        if self.fills:
            total_value = sum(fill.price * fill.size for fill in self.fills)
            total_size = self.get_filled_size()
            return total_value / total_size
        return None

    def get_last_fill_price(self) -> Optional[ExtendedDecimal]:
        """Get the price of the last fill."""
        if self.fills:
            return self.fills[-1].price
        return None

    def get_last_fill_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the last fill."""
        if self.fills:
            return self.fills[-1].timestamp
        return None

    def _apply_slippage(self, price: ExtendedDecimal) -> ExtendedDecimal:
        """Apply slippage to the given price if slippage is specified in the order details."""
        if self.details.slippage is not None:
            slippage_factor = ExtendedDecimal("1") + (
                self.details.slippage * self.details.direction.value
            )
            return price * slippage_factor
        return price

    def update_trailing_stop(self, current_price: ExtendedDecimal):
        """Update the trailing stop price based on the current market price."""
        if self.details.exectype != self.ExecType.TRAILING:
            logger_main.log_and_raise(
                ValueError("This method is only applicable for trailing stop orders")
            )

        is_long = self.details.direction == self.Direction.LONG
        if self.trailing_activation_price is None:
            self.trailing_activation_price = current_price
            self.details.price = self._calculate_trailing_stop_price(
                current_price, is_long
            )
        elif (is_long and current_price > self.trailing_activation_price) or (
            not is_long and current_price < self.trailing_activation_price
        ):
            self.trailing_activation_price = current_price
            self.details.price = self._calculate_trailing_stop_price(
                current_price, is_long
            )


class OrderManager:
    """
    Manages the creation, modification, and cancellation of orders and order groups.
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_groups: Dict[str, OrderGroup] = {}

    def create_order(self, details: OrderDetails) -> Order:
        """
        Create a new order.

        Args:
            details (OrderDetails): The details of the order to be created.

        Returns:
            Order: The newly created order.
        """
        order_id = str(uuid.uuid4())
        order = Order(order_id, details)
        self.orders[order_id] = order
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

    def create_bracket_group(
        self, entry_order: Order, take_profit_order: Order, stop_loss_order: Order
    ) -> BracketGroup:
        """
        Create a new Bracket order group.

        Args:
            entry_order (Order): The entry order for the bracket.
            take_profit_order (Order): The take-profit order for the bracket.
            stop_loss_order (Order): The stop-loss order for the bracket.

        Returns:
            BracketGroup: The newly created Bracket group.
        """
        bracket_group = BracketGroup()
        bracket_group.add_order(entry_order)
        bracket_group.add_order(take_profit_order)
        bracket_group.add_order(stop_loss_order)
        self.order_groups[bracket_group.id] = bracket_group
        logger_main.info(f"Created Bracket group: {bracket_group.id}")
        return bracket_group

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an order.

        Args:
            order_id (str): The ID of the order to be cancelled.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.cancel()
            logger_main.info(f"Cancelled order: {order_id}")
        else:
            logger_main.warning(f"Order not found: {order_id}")

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

        Side effects:
            - Updates order statuses
            - Removes filled or cancelled orders from pending orders
            - Triggers order group events for filled orders
        """
        processed_orders = []

        for order in list(self.orders.values()):
            if not order.is_active:
                continue

            symbol = order.details.ticker
            timeframe = order.details.timeframe or min(market_data[symbol].keys())

            try:
                current_bar = market_data[symbol][timeframe]
            except KeyError:
                logger_main.warning(
                    f"No market data for {symbol} at timeframe {timeframe}. Skipping order {order.id}."
                )
                continue

            is_filled, fill_price = order.is_filled(current_bar)

            if is_filled:
                order.on_fill(order.get_remaining_size(), fill_price, timestamp)
                processed_orders.append(order)

                if order.order_group:
                    order.order_group.on_order_filled(order)

            elif order.is_expired(timestamp):
                order.on_cancel()
                processed_orders.append(order)

        self.cleanup_completed_orders_and_groups()

        return processed_orders

    def update(self, timestamp: datetime) -> None:
        """
        Update the OrderManager state in each cycle.

        This method performs housekeeping tasks for the OrderManager, such as:
        - Checking for and handling expired orders
        - Updating trailing stop orders
        - Cleaning up completed orders and groups

        Args:
            timestamp (datetime): The current timestamp.

        Side effects:
            - May cancel expired orders
            - Updates trailing stop orders
            - Removes completed orders and groups
        """
        for order in list(self.orders.values()):
            if not order.is_active:
                continue

            if order.is_expired(timestamp):
                self.cancel_order(order.id)

            elif order.details.exectype == Order.ExecType.TRAILING:
                # Update trailing stop price
                current_price = self._get_current_price(order.details.ticker)
                order.update_trailing_stop(current_price)

        self.cleanup_completed_orders_and_groups()

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
        else:
            logger_main.warning(f"Fill event received for unknown order: {order_id}")

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
        else:
            logger_main.warning(f"Reject event received for unknown order: {order_id}")

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

    def __repr__(self) -> str:
        """Return a string representation of the OrderManager."""
        return f"OrderManager(active_orders={len(self.get_active_orders())}, active_groups={len(self.get_active_groups())})"
