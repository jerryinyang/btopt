from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from .data.bar import Bar
from .log_config import logger_main
from .order import Order
from .util.ext_decimal import ExtendedDecimal


@dataclass
class TradeMetrics:
    """Dataclass to store various metrics related to a trade."""

    pnl: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    pnl_percent: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    realized_pnl: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    unrealized_pnl: ExtendedDecimal = field(
        default_factory=lambda: ExtendedDecimal("0")
    )
    commission: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    slippage: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    max_runup: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    max_runup_percent: ExtendedDecimal = field(
        default_factory=lambda: ExtendedDecimal("0")
    )
    max_drawdown: ExtendedDecimal = field(default_factory=lambda: ExtendedDecimal("0"))
    max_drawdown_percent: ExtendedDecimal = field(
        default_factory=lambda: ExtendedDecimal("0")
    )


class Trade:
    """Represents a trading position, including entry and exit information, and various metrics."""

    class Status(Enum):
        """Enum representing the current status of a trade."""

        ACTIVE = "Active"
        PARTIALLY_CLOSED = "Partially Closed"
        CLOSED = "Closed"

    def __init__(
        self,
        trade_id: int,
        entry_order: Order,
        entry_bar: Bar,
        commission_rate: Optional[ExtendedDecimal] = None,
        strategy_id: Optional[str] = None,
    ):
        """Initialize a new Trade instance."""
        self.id: int = trade_id
        self.ticker: str = entry_order.details.ticker
        self.direction: Order.Direction = entry_order.details.direction
        self.initial_size: ExtendedDecimal = entry_order.details.size
        self.current_size: ExtendedDecimal = ExtendedDecimal("0")

        self.entry_orders: List[Order] = [entry_order]
        self.exit_orders: List[Order] = []

        self.entry_price: ExtendedDecimal = ExtendedDecimal("0")
        self.entry_timestamp: datetime = entry_bar.timestamp
        self.entry_bar: Bar = entry_bar

        self.exit_price: Optional[ExtendedDecimal] = None
        self.exit_timestamp: Optional[datetime] = None
        self.exit_bar: Optional[Bar] = None

        self.parent_id: Optional[int] = entry_order.details.parent_id
        self.status: Trade.Status = Trade.Status.ACTIVE
        self.metrics: TradeMetrics = TradeMetrics()
        self.alpha_name: Optional[str] = getattr(
            entry_order.details, "alpha_name", None
        )

        self.commission_rate: Optional[ExtendedDecimal] = commission_rate
        self.strategy_id: Optional[str] = strategy_id or entry_order.details.strategy_id

        self._process_entry_order(entry_order)

    def _process_entry_order(self, order: Order):
        """Process an entry order, updating the trade's size and average entry price."""
        filled_size = order.get_filled_size()
        fill_price = order.get_average_fill_price()

        if fill_price is None:
            logger_main.log_and_raise(ValueError("Entry order has no fill price"))

        new_size = self.current_size + filled_size
        self.entry_price = (
            self.entry_price * self.current_size + fill_price * filled_size
        ) / new_size
        self.current_size = new_size

        self._calculate_entry_commission_and_slippage(order)

    def _calculate_entry_commission_and_slippage(self, order: Order):
        """Calculates and sets the commission and slippage for an entry order."""
        if self.commission_rate:
            self.metrics.commission += (
                order.get_filled_size()
                * order.get_average_fill_price()
                * self.commission_rate
            )

        if order.details.slippage:
            self.metrics.slippage += (
                abs(order.details.price - order.get_average_fill_price())
                * order.get_filled_size()
            )

    def add_entry(self, order: Order):
        """Add an additional entry order to the trade (e.g., for pyramiding)."""
        self.entry_orders.append(order)
        self._process_entry_order(order)

    def update(self, current_bar: Bar) -> None:
        """Updates the trade metrics based on the current market price."""
        current_price = current_bar.close
        self.metrics.unrealized_pnl = self._calculate_pnl(current_price)
        self.metrics.pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        self.metrics.pnl_percent = self._calculate_pnl_percent(self.metrics.pnl)

        self._update_runup_drawdown(self.metrics.pnl, self.metrics.pnl_percent)

    def close(
        self, exit_order: Order, exit_bar: Bar, size: Optional[ExtendedDecimal] = None
    ) -> None:
        """
        Close the trade (fully or partially) with the given exit order and update relevant information.

        Args:
            exit_order (Order): The order used to close the trade.
            exit_bar (Bar): The bar at which the trade is being closed.
            size (Optional[ExtendedDecimal]): The size to close. If None, closes the entire trade.
        """
        if exit_order.get_filled_size() > self.current_size:
            logger_main.warning(
                f"Exit order size ({exit_order.get_filled_size()}) exceeds current trade size ({self.current_size}). Closing entire trade."
            )
            size = self.current_size
        else:
            size = size or self.current_size

        exit_price = exit_order.get_average_fill_price()
        if exit_price is None:
            logger_main.log_and_raise(ValueError("Exit order has no fill price"))

        # Calculate P&L for this exit
        exit_pnl = self._calculate_pnl(exit_price, size)
        self.metrics.realized_pnl += exit_pnl

        # Calculate and add commission and slippage for this exit
        if self.commission_rate:
            exit_commission = exit_price * size * self.commission_rate
            self.metrics.commission += exit_commission

        if exit_order.details.slippage:
            exit_slippage = abs(exit_order.details.price - exit_price) * size
            self.metrics.slippage += exit_slippage

        # Update trade size
        self.current_size -= size

        # Update trade status
        if self.current_size == ExtendedDecimal("0"):
            self.status = self.Status.CLOSED
            self.exit_price = exit_price
            self.exit_timestamp = exit_order.fill_timestamp or exit_bar.timestamp
            self.exit_bar = exit_bar
        else:
            self.status = self.Status.PARTIALLY_CLOSED

        self._finalize_metrics()

        logger_main.info(
            f"Trade {self.id} {'closed' if self.status == self.Status.CLOSED else 'partially closed'}: "
            f"exit price {exit_price}, size {size}, remaining size {self.current_size}"
        )

    def reverse(self, order: Order, bar: Bar) -> None:
        """
        Reverse the trade's direction and update relevant attributes.

        This method closes the current trade and opens a new one in the opposite direction.

        Args:
            order (Order): The order that triggered the reversal.
            bar (Bar): The current price bar.
        """
        # Close the current trade
        self.close(order, bar)

        # Reverse the direction
        self.direction = (
            Order.Direction.LONG
            if self.direction == Order.Direction.SHORT
            else Order.Direction.SHORT
        )

        # Update trade attributes
        self.entry_price = order.get_average_fill_price()
        self.entry_timestamp = bar.timestamp
        self.entry_bar = bar
        self.initial_size = order.get_filled_size()
        self.current_size = self.initial_size
        self.status = self.Status.ACTIVE

        # Reset exit information
        self.exit_price = None
        self.exit_timestamp = None
        self.exit_bar = None

        # Reset metrics
        self.metrics = TradeMetrics()

        # Log the reversal
        logger_main.info(
            f"Trade {self.id} reversed: new direction {self.direction.name}, size {self.current_size}"
        )

    def _calculate_pnl(
        self, current_price: ExtendedDecimal, size: Optional[ExtendedDecimal] = None
    ) -> ExtendedDecimal:
        """Calculates the profit/loss for the trade."""
        size = size or self.current_size
        price_diff = (current_price - self.entry_price) * self.direction.value
        return price_diff * size

    def _calculate_pnl_percent(self, pnl: ExtendedDecimal) -> ExtendedDecimal:
        """Calculates the profit/loss percentage for the trade."""
        if self.initial_size == ExtendedDecimal("0"):
            return ExtendedDecimal("0")
        return (pnl / (self.entry_price * self.initial_size)) * ExtendedDecimal("100")

    def _update_runup_drawdown(
        self, pnl: ExtendedDecimal, pnl_percent: ExtendedDecimal
    ) -> None:
        """Updates the maximum runup and drawdown for the trade."""
        if pnl > self.metrics.max_runup:
            self.metrics.max_runup = pnl
            self.metrics.max_runup_percent = pnl_percent
        elif pnl < self.metrics.max_drawdown:
            self.metrics.max_drawdown = pnl
            self.metrics.max_drawdown_percent = pnl_percent

    def _finalize_metrics(self) -> None:
        """Finalizes the trade metrics upon closing or partial closing."""
        self.metrics.pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        self.metrics.pnl_percent = self._calculate_pnl_percent(self.metrics.pnl)

    def duration(self) -> timedelta:
        """Calculates the duration of the trade."""
        end_time = self.exit_timestamp or datetime.now()
        return end_time - self.entry_timestamp

    def get_average_entry_price(self) -> ExtendedDecimal:
        """Get the average entry price of the trade."""
        return self.entry_price

    def get_realized_pnl(self) -> ExtendedDecimal:
        """Get the realized PNL of the trade."""
        return self.metrics.realized_pnl

    def get_unrealized_pnl(self) -> ExtendedDecimal:
        """Get the unrealized PNL of the trade."""
        return self.metrics.unrealized_pnl

    def __repr__(self) -> str:
        """Returns a string representation of the Trade object."""
        return (
            f"Trade(id={self.id}, ticker={self.ticker}, direction={self.direction.name}, "
            f"initial_size={self.initial_size}, current_size={self.current_size}, "
            f"entry_price={self.entry_price}, status={self.status.name}, "
            f"realized_pnl={self.metrics.realized_pnl:.2f}, unrealized_pnl={self.metrics.unrealized_pnl:.2f})"
        )

    def to_dict(self) -> dict:
        """Converts the Trade object to a dictionary for serialization."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "direction": self.direction.name,
            "initial_size": str(self.initial_size),
            "current_size": str(self.current_size),
            "entry_price": str(self.entry_price),
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_timestamp": self.exit_timestamp.isoformat()
            if self.exit_timestamp
            else None,
            "status": self.status.name,
            "realized_pnl": str(self.metrics.realized_pnl),
            "unrealized_pnl": str(self.metrics.unrealized_pnl),
            "pnl": str(self.metrics.pnl),
            "pnl_percent": str(self.metrics.pnl_percent),
            "commission": str(self.metrics.commission),
            "slippage": str(self.metrics.slippage),
            "max_runup": str(self.metrics.max_runup),
            "max_runup_percent": str(self.metrics.max_runup_percent),
            "max_drawdown": str(self.metrics.max_drawdown),
            "max_drawdown_percent": str(self.metrics.max_drawdown_percent),
            "alpha_name": self.alpha_name,
            "strategy_id": self.strategy_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        """Creates a Trade object from a dictionary representation."""
        entry_order = Order(
            order_id=data["id"],
            details=Order.OrderDetails(
                ticker=data["ticker"],
                direction=Order.Direction[data["direction"]],
                size=ExtendedDecimal(data["initial_size"]),
                price=ExtendedDecimal(data["entry_price"]),
                exectype=Order.ExecType.MARKET,
                timestamp=datetime.fromisoformat(data["entry_timestamp"]),
                timeframe=None,  # Timeframe information is not stored in the dict
                strategy_id=data.get("strategy_id"),
            ),
        )
        entry_bar = Bar(
            open=ExtendedDecimal(data["entry_price"]),
            high=ExtendedDecimal(data["entry_price"]),
            low=ExtendedDecimal(data["entry_price"]),
            close=ExtendedDecimal(data["entry_price"]),
            volume=0,
            timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            timeframe=None,  # Timeframe information is not stored in the dict
            ticker=data["ticker"],
        )
        trade = cls(
            data["id"], entry_order, entry_bar, strategy_id=data.get("strategy_id")
        )
        trade.current_size = ExtendedDecimal(data["current_size"])
        trade.metrics = TradeMetrics(
            realized_pnl=ExtendedDecimal(data["realized_pnl"]),
            unrealized_pnl=ExtendedDecimal(data["unrealized_pnl"]),
            pnl=ExtendedDecimal(data["pnl"]),
            pnl_percent=ExtendedDecimal(data["pnl_percent"]),
            commission=ExtendedDecimal(data["commission"]),
            slippage=ExtendedDecimal(data["slippage"]),
            max_runup=ExtendedDecimal(data["max_runup"]),
            max_runup_percent=ExtendedDecimal(data["max_runup_percent"]),
            max_drawdown=ExtendedDecimal(data["max_drawdown"]),
            max_drawdown_percent=ExtendedDecimal(data["max_drawdown_percent"]),
        )
        trade.status = Trade.Status[data["status"]]
        trade.alpha_name = data["alpha_name"]
        if data["exit_price"]:
            trade.exit_price = ExtendedDecimal(data["exit_price"])
            trade.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"])
        return trade
