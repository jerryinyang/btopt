import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from .data.bar import Bar
from .data.timeframe import Timeframe
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


class OrderGroupType(Enum):
    OCO = "One-Cancels-the-Other"
    OCA = "One-Cancels-All"
    BRACKET = "Bracket"


class OrderGroup(ABC):
    def __init__(self, group_type: OrderGroupType):
        self.id: str = str(uuid.uuid4())
        self.type: OrderGroupType = group_type
        self.orders: List[Order] = []
        self.active: bool = False

    @abstractmethod
    def add_order(self, order: "Order") -> None:
        pass

    @abstractmethod
    def remove_order(self, order: "Order") -> None:
        pass

    @abstractmethod
    def on_order_filled(self, filled_order: "Order") -> None:
        pass

    @abstractmethod
    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        pass

    @abstractmethod
    def on_order_rejected(self, rejected_order: "Order") -> None:
        pass

    def activate(self) -> None:
        """Activate the order group and all its child orders."""
        self.active = True
        for order in self.orders:
            order.activate()

    def deactivate(self) -> None:
        """Deactivate the order group and all its child orders."""
        self.active = False
        for order in self.orders:
            order.deactivate()

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
        return "Active" if self.active else "Inactive"


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
    class Role(Enum):
        """Enum class for different roles in the bracket order"""

        ENTRY = "Entry"
        LIMIT = "Limit"
        STOP = "Stop"

        # Adding aliases
        PARENT = PRIMARY = ENTRY
        TP = TAKE_PROFIT = LIMIT
        SL = STOP_LOSS = STOP

    def __init__(self):
        super().__init__(OrderGroupType.BRACKET)
        self.entry_order: Optional[Order] = None
        self.take_profit_order: Optional[Order] = None
        self.stop_loss_order: Optional[Order] = None

    def add_order(self, order: "Order", role: "BracketGroup.Role") -> None:
        """Add an order to the bracket group."""

        if role == BracketGroup.Role.ENTRY:
            if self.entry_order:
                logger_main.warning(
                    f"Entry Order `{self.entry_order}` for Bracket Order `{self.id}` already exists. Order {order.id} will not be added."
                )

            self.entry_order = order

        elif role == BracketGroup.Role.LIMIT:
            if self.take_profit_order:
                logger_main.warning(
                    f"Limit Order `{self.take_profit_order}` for Bracket Order `{self.id}` already exists. Order {order.id} will not be added."
                )

            # Confirm that the order exectype is correct
            elif order.details.exectype in [
                Order.ExecType.LIMIT,
                Order.ExecType.EXIT_LIMIT,
            ]:
                self.take_profit_order = order

            else:
                logger_main.log_and_raise(
                    f"Invalid order exectype for the Bracket Limit Order : {order.details.exectype}. Expected exectypes are [Order.ExecType.LIMIT, Order.ExecType.EXIT_LIMIT]."
                )

        elif role == BracketGroup.Role.STOP:
            if self.stop_loss_order:
                logger_main.warning(
                    f"Stop Order `{self.stop_loss_order}` for Bracket Order `{self.id}` already exists. Order {order.id} will not be added."
                )

            elif order.details.exectype in [
                Order.ExecType.STOP,
                Order.ExecType.STOP_LIMIT,
                Order.ExecType.EXIT_STOP,
            ]:
                self.stop_loss_order = order

            else:
                logger_main.log_and_raise(
                    f"Invalid order exectype for the Bracket Stop Order : {order.details.exectype}. Expected exectypes are [Order.ExecType.STOP, Order.ExecType.STOP_LIMIT, Order.ExecType.EXIT_STOP]."
                )

        else:
            logger_main.warning(
                f"Invalid Bracket Role value passed for the order: {role}"
            )

        self.orders.append(order)
        order.order_group = self

    def remove_order(self, order: "Order") -> None:
        """Remove an order from the bracket group."""
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
        """Handle the event when an order in the group is filled."""
        if filled_order == self.entry_order:
            self.activate()
        elif filled_order in [self.take_profit_order, self.stop_loss_order]:
            self.deactivate()
            other_exit_order = (
                self.take_profit_order
                if filled_order == self.stop_loss_order
                else self.stop_loss_order
            )
            if other_exit_order and other_exit_order.status != Order.Status.FILLED:
                other_exit_order.cancel()

    def on_order_cancelled(self, cancelled_order: "Order") -> None:
        """Handle the event when an order in the group is cancelled."""
        if cancelled_order == self.entry_order:
            self.deactivate()

    def on_order_rejected(self, rejected_order: "Order") -> None:
        """Handle the event when an order in the group is rejected."""
        if rejected_order == self.entry_order:
            self.deactivate()

    def activate(self) -> None:
        """Activate the bracket order group."""
        super().activate()
        if self.take_profit_order:
            self.take_profit_order.activate()
        if self.stop_loss_order:
            self.stop_loss_order.activate()

    def deactivate(self) -> None:
        """Deactivate the bracket order group."""
        super().deactivate()
        if self.take_profit_order:
            self.take_profit_order.deactivate()
        if self.stop_loss_order:
            self.stop_loss_order.deactivate()


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
    stoplimit_price: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    trailing_percent: Optional[ExtendedDecimal] = field(default_factory=lambda: None)
    slippage: Optional[ExtendedDecimal] = field(default_factory=lambda: None)


@dataclass
class OCOOrderDetails:
    limit_order: OrderDetails
    stop_order: OrderDetails


@dataclass
class OCAOrderDetails:
    orders: List[OrderDetails]


@dataclass
class BracketOrderDetails:
    entry_order: OrderDetails
    take_profit_order: Optional[OrderDetails] = None
    stop_loss_order: Optional[OrderDetails] = None


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
        self.execution_bar_timestamp: Optional[datetime] = None

    def is_filled(self, bar: Bar) -> Tuple[bool, Optional[ExtendedDecimal]]:
        """
        Check if the order is filled based on the current price bar.

        This method determines whether the order should be filled given the current
        market conditions represented by the price bar. It also calculates and returns
        the fill price, taking into account any specified slippage.

        Args:
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[ExtendedDecimal]]: A tuple containing a boolean indicating
            whether the order is filled, and the fill price if filled (None otherwise).
        """
        if self.status == self.Status.FILLED:
            return True, self.get_last_fill_price()

        filled, price = self._check_fill_conditions(bar)
        if filled:
            fill_price = self._apply_slippage(price)
            return True, fill_price
        return False, None

    def _check_fill_conditions(
        self, bar: Bar
    ) -> Tuple[bool, Optional[ExtendedDecimal]]:
        """
        Check if the order's fill conditions are met based on the current price bar.

        This method implements the logic for determining whether an order should be
        filled based on its type (market, limit, stop, etc.) and the current market conditions.

        Args:
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[ExtendedDecimal]]: A tuple containing a boolean indicating
            whether the fill conditions are met, and the potential fill price if conditions
            are met (None otherwise).
        """
        if self.details.exectype == self.ExecType.MARKET:
            return True, bar.open

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

        if self.execution_bar_timestamp is None:
            self.execution_bar_timestamp = timestamp

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

    def deactivate(self) -> None:
        """
        Deactivate the order.

        This method sets the order's active status to False, effectively removing it from consideration
        for execution without cancelling it.
        """
        self.is_active = False
        logger_main.info(f"Order {self.id} deactivated")

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
        """
        Apply slippage to the given price if slippage is specified in the order details.

        Args:
            price (ExtendedDecimal): The original price before slippage.

        Returns:
            ExtendedDecimal: The price after applying slippage.
        """
        if self.details.slippage is not None:
            slippage_factor = ExtendedDecimal("1") + (
                self.details.slippage * self.details.direction.value
            )
            return price * slippage_factor
        return price

    def sort_key(self) -> Tuple[int, ExtendedDecimal]:
        """
        Generate a sort key for the order based on its type and price.

        This method is used to determine the order's priority when multiple orders
        are being processed. It takes into account the order type and the current
        market direction.

        Returns:
            Tuple[int, ExtendedDecimal]: A tuple used as a sort key, where the first
            element is an integer representing the order type priority, and the second
            is the order price.
        """
        if self.details.exectype == self.ExecType.MARKET:
            return (0, ExtendedDecimal("0"))
        elif self.details.exectype in [self.ExecType.STOP, self.ExecType.STOP_LIMIT]:
            return (1, self.details.price)
        else:  # LIMIT orders
            return (2, self.details.price)

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

    def __repr__(self) -> str:
        """Return a string representation of the Order."""
        return (
            f"Order(id={self.id}, ticker={self.details.ticker}, "
            f"direction={self.details.direction.name}, "
            f"price={self.details.price}, status={self.status.name}, "
            f"exectype={self.details.exectype.name}, "
            f"filled={self.get_filled_size()}/{self.details.size})"
        )

    def __lt__(self, other: "Order") -> bool:
        """
        Compare orders for sorting based on creation time.

        Args:
            other (Order): The other order to compare with.

        Returns:
            bool: True if this order should be executed before the other order, False otherwise.
        """
        if self.details.exectype != other.details.exectype:
            # Market orders always come first
            return self.details.exectype == self.ExecType.MARKET
        elif self.details.exectype in [
            self.ExecType.LIMIT,
            self.ExecType.STOP,
            self.ExecType.STOP_LIMIT,
        ]:
            # For limit and stop orders, sort by price (ascending for buy, descending for sell)
            if self.details.direction == other.details.direction:
                if self.details.direction == self.Direction.LONG:
                    return self.details.price > other.details.price
                else:
                    return self.details.price < other.details.price
            else:
                # If directions are different, maintain FIFO order
                return self.details.timestamp < other.details.timestamp
        else:
            # For other order types, maintain FIFO order
            return self.details.timestamp < other.details.timestamp
