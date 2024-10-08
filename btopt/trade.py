from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from .data.bar import Bar
from .order import Order
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


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
        self.initial_size: ExtendedDecimal = entry_order.get_filled_size()
        self.current_size: ExtendedDecimal = self.initial_size

        self.entry_order: Order = entry_order
        self.exit_orders: List[Order] = []

        self.entry_price: ExtendedDecimal = entry_order.get_last_fill_price()
        self.entry_timestamp: datetime = (
            entry_order.get_last_fill_timestamp() or entry_bar.timestamp
        )
        self.execution_bar_timestamp: datetime = entry_bar.timestamp
        self.entry_bar: Bar = entry_bar

        self.exit_price: Optional[ExtendedDecimal] = None
        self.exit_timestamp: Optional[datetime] = None
        self.exit_bar: Optional[Bar] = None

        self.status: Trade.Status = Trade.Status.ACTIVE
        self.metrics: TradeMetrics = TradeMetrics()

        self.commission_rate: Optional[ExtendedDecimal] = commission_rate
        self.strategy_id: Optional[str] = strategy_id or entry_order.details.strategy_id

        if not self.strategy_id:
            logger_main.log_and_raise(
                "strategy_id cannot be of NoneType for Trade objects."
            )

        self._calculate_entry_commission_and_slippage(entry_order)

    def _calculate_entry_commission_and_slippage(self, order: Order):
        """Calculates and sets the commission and slippage for the entry order."""
        if self.commission_rate:
            self.metrics.commission = (
                order.get_filled_size()
                * order.get_last_fill_price()
                * self.commission_rate
            )

        if order.details.slippage:
            self.metrics.slippage = (
                abs(order.details.price - order.get_last_fill_price())
                * order.get_filled_size()
            )

    def update(self, current_bar: Bar) -> None:
        """Updates the trade metrics based on the current market price."""
        current_price = current_bar.close
        self.metrics.unrealized_pnl = self._calculate_pnl(current_price)
        self.metrics.pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        self.metrics.pnl_percent = self._calculate_pnl_percent(self.metrics.pnl)

        self._update_runup_drawdown(self.metrics.pnl, self.metrics.pnl_percent)

    def close(
        self,
        exit_order: Order,
        exit_price: ExtendedDecimal,
        exit_bar: Bar,
        size: Optional[ExtendedDecimal] = None,
    ) -> None:
        """
        Close the trade (fully or partially) and update relevant information.

        This method handles the closing process of a trade, including calculation of P&L,
        commissions, and slippage. It updates the trade's status and metrics accordingly.

        Args:
            exit_order (Order): The order used to close the trade.
            exit_price (ExtendedDecimal): The price at which the trade is being closed.
            exit_bar (Bar): The bar at which the trade is being closed.
            size (Optional[ExtendedDecimal]): The size to close. If None, closes the entire trade.

        Raises:
            ValueError: If the exit order size exceeds the current trade size or if the exit price is invalid.

        Note:
            This method directly modifies the trade's attributes and metrics.
        """
        if size is not None and size > self.current_size:
            logger_main.warning(
                f"Exit size ({size}) exceeds current trade size ({self.current_size}). Closing entire trade."
            )
            size = self.current_size
        else:
            size = size or self.current_size

        if exit_price <= ExtendedDecimal("0"):
            logger_main.log_and_raise(ValueError("Invalid exit price"))

        # Check if the exit is allowed on this bar
        if exit_bar.timestamp <= self.execution_bar_timestamp:
            # Apply bar formation assumption algorithm
            if not self._valid_same_bar_exit(exit_price, exit_bar):
                logger_main.warning(
                    f"Attempted to close trade {self.id} on the same bar it was created, and doesn't meet the bar-formation condition. Skipping closure."
                )
                return

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

        # Update trade status and exit information
        if self.current_size == ExtendedDecimal("0"):
            self.status = self.Status.CLOSED
            self.exit_price = exit_order.get_average_fill_price() or exit_price
            self.exit_timestamp = (
                exit_order.get_last_fill_timestamp() or exit_bar.timestamp
            )
            self.exit_bar = exit_bar
        else:
            self.status = self.Status.PARTIALLY_CLOSED

        self.exit_orders.append(exit_order)
        self._finalize_metrics()

        logger_main.info(
            f"Trade {self.id} {'closed' if self.status == self.Status.CLOSED else 'partially closed'}: "
            f"exit price {exit_price}, size {size}, remaining size {self.current_size}"
        )

    def _valid_same_bar_exit(self, exit_price: ExtendedDecimal, bar: Bar):
        return (bar.close > bar.open and exit_price > self.entry_price) or (
            bar.close < bar.open and exit_price < self.entry_price
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
        self.metrics.unrealized_pnl = ExtendedDecimal("0")
        self.metrics.pnl = self.metrics.realized_pnl
        self.metrics.pnl_percent = self._calculate_pnl_percent(self.metrics.pnl)
        self._update_runup_drawdown(self.metrics.pnl, self.metrics.pnl_percent)

    def duration(self) -> timedelta:
        """Calculates the duration of the trade."""
        end_time = self.exit_timestamp  # or datetime.now()
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
