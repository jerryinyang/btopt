from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from .data.bar import Bar
from .data.timeframe import Timeframe
from .log_config import logger_main


@dataclass(frozen=True)
class OrderDetails:
    """A frozen dataclass representing the details of an order."""

    ticker: str
    direction: "Order.Direction"
    size: Decimal
    price: Decimal
    exectype: "Order.ExecType"
    timestamp: datetime
    timeframe: Timeframe
    expiry: Optional[datetime] = None
    stoplimit_price: Optional[Decimal] = None
    parent_id: Optional[int] = None
    exit_profit: Optional[Decimal] = None
    exit_loss: Optional[Decimal] = None
    exit_profit_percent: Optional[Decimal] = None
    exit_loss_percent: Optional[Decimal] = None
    trailing_percent: Optional[Decimal] = None
    slippage: Optional[Decimal] = None
    strategy_id: Optional[str] = None


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

    class FamilyRole(Enum):
        """Enum representing the role of an order in a family of related orders."""

        PARENT = "Parent"
        CHILD_EXIT = "ChildExit"
        CHILD_REDUCE = "ChildReduce"

    def __init__(self, order_id: int, details: OrderDetails):
        """Initialize a new Order instance."""
        self.id = order_id
        self.details = details
        self.status = self.Status.CREATED
        self.fill_price: Optional[Decimal] = None
        self.fill_timestamp: Optional[datetime] = None
        self.filled_size: Decimal = Decimal("0")
        self.children: List["Order"] = []
        self.family_role = (
            self.FamilyRole.PARENT
            if self.details.parent_id is None
            else self.FamilyRole.CHILD_EXIT
        )
        self.trailing_activation_price: Optional[Decimal] = None

        self._create_child_orders()

    def _create_child_orders(self):
        """Create child orders for exit strategies (profit target and stop loss)."""
        if self.details.exit_profit or self.details.exit_profit_percent:
            self._add_child_order(
                self.ExecType.EXIT_LIMIT,
                self.details.exit_profit,
                self.details.exit_profit_percent,
            )

        if self.details.exit_loss or self.details.exit_loss_percent:
            self._add_child_order(
                self.ExecType.EXIT_STOP,
                self.details.exit_loss,
                self.details.exit_loss_percent,
            )

    def _add_child_order(
        self, exectype: ExecType, price: Optional[Decimal], percent: Optional[Decimal]
    ):
        """Add a child order with the specified execution type and price or percentage."""
        child_details = OrderDetails(
            ticker=self.details.ticker,
            direction=self.Direction.SHORT
            if self.details.direction == self.Direction.LONG
            else self.Direction.LONG,
            size=self.details.size,
            price=price
            if price is not None
            else self._calculate_price_from_percent(percent),
            exectype=exectype,
            timestamp=self.details.timestamp,
            timeframe=self.details.timeframe,
            parent_id=self.id,
            slippage=self.details.slippage,
        )
        child_order = Order(order_id=hash(child_details), details=child_details)
        child_order.family_role = self.FamilyRole.CHILD_EXIT
        self.children.append(child_order)

    def _calculate_price_from_percent(self, percent: Decimal) -> Decimal:
        """Calculate the price based on a percentage difference from the parent order's price."""
        if percent is None:
            logger_main.log_and_raise(
                ValueError("Percent value is required for calculation")
            )

        return self.details.price * (
            Decimal("1") + percent * self.details.direction.value
        )

    def is_filled(self, bar: "Bar") -> tuple[bool, Optional[Decimal]]:
        """Check if the order is filled based on the current price bar."""
        if self.status == self.Status.FILLED:
            return True, self.fill_price

        filled, price = self._check_fill_conditions(bar)
        if filled:
            self.fill(price, bar.timestamp)
        return filled, price

    def _check_fill_conditions(self, bar: "Bar") -> tuple[bool, Optional[Decimal]]:
        """Check if the order's fill conditions are met based on the current price bar."""
        if self.details.exectype == self.ExecType.MARKET:
            return True, self._apply_slippage(bar.open)

        if self.details.direction == self.Direction.LONG:
            return self._check_long_fill_conditions(bar)
        else:
            return self._check_short_fill_conditions(bar)

    def _check_long_fill_conditions(self, bar: "Bar") -> tuple[bool, Optional[Decimal]]:
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
    ) -> tuple[bool, Optional[Decimal]]:
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
    ) -> tuple[bool, Optional[Decimal]]:
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
        self, activation_price: Decimal, is_long: bool
    ) -> Decimal:
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
    ) -> tuple[bool, Optional[Decimal]]:
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

    def _apply_slippage(self, price: Decimal) -> Decimal:
        """Apply slippage to the given price if slippage is specified in the order details."""
        if self.details.slippage is not None:
            slippage_factor = Decimal("1") + (
                self.details.slippage * self.details.direction.value
            )
            return price * slippage_factor
        return price

    def fill(self, price: Decimal, timestamp: datetime, size: Optional[Decimal] = None):
        """Mark the order as filled (fully or partially) at the specified price and timestamp."""
        fill_size = size or (self.details.size - self.filled_size)
        self.filled_size += fill_size
        self.fill_price = (self.fill_price or Decimal("0")) + price * fill_size
        self.fill_timestamp = timestamp

        if self.filled_size >= self.details.size:
            self.status = self.Status.FILLED
        elif self.filled_size > Decimal("0"):
            self.status = self.Status.PARTIALLY_FILLED

    def cancel(self):
        """Cancel the order and all its child orders."""
        if self.status not in [self.Status.FILLED, self.Status.CANCELED]:
            self.status = self.Status.CANCELED
            for child in self.children:
                child.cancel()

    def is_active(self) -> bool:
        """Check if the order is currently active."""
        return self.status in [
            self.Status.CREATED,
            self.Status.ACCEPTED,
            self.Status.PARTIALLY_FILLED,
        ]

    def is_expired(self, current_time: datetime) -> bool:
        """Check if the order has expired."""
        return self.details.expiry is not None and current_time >= self.details.expiry

    def update_trailing_stop(self, current_price: Decimal):
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

    def is_reduce_only(self) -> bool:
        """Check if this order is meant to reduce an existing position."""
        return self.family_role in [
            self.FamilyRole.CHILD_EXIT,
            self.FamilyRole.CHILD_REDUCE,
        ]

    def get_filled_size(self) -> Decimal:
        """Get the total filled size of the order."""
        return self.filled_size

    def get_remaining_size(self) -> Decimal:
        """Get the remaining unfilled size of the order."""
        return self.details.size - self.filled_size

    def get_average_fill_price(self) -> Optional[Decimal]:
        """Get the average fill price of the order."""
        if self.filled_size > Decimal("0"):
            return self.fill_price / self.filled_size
        return None

    def update_size(self, new_size: Decimal):
        """Update the order size, ensuring it doesn't decrease below the filled size."""
        if new_size < self.filled_size:
            logger_main.log_and_raise(
                ValueError("New size cannot be less than the filled size")
            )
        self.details.size = new_size

    def __eq__(self, other):
        """Check if two orders are equal based on their IDs."""
        if not isinstance(other, Order):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        """Generate a hash value for the order based on its ID."""
        return hash(self.id)

    def __repr__(self):
        """Return a string representation of the order."""
        return (
            f"Order(id={self.id}, ticker={self.details.ticker}, "
            f"direction={self.details.direction.name}, "
            f"price={self.details.price}, status={self.status.name}, "
            f"exectype={self.details.exectype.name}, "
            f"filled={self.filled_size}/{self.details.size})"
        )
