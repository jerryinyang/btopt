from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from .data.bar import Bar
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails


@dataclass
class TradeMetrics:
    """
    Dataclass to store various metrics related to a trade.

    Attributes:
        pnl (Decimal): Profit and Loss of the trade.
        pnl_percent (Decimal): Percentage of Profit and Loss relative to the trade size.
        commission (Decimal): Commission paid for the trade.
        slippage (Decimal): Slippage cost for the trade.
        max_runup (Decimal): Maximum unrealized profit during the trade.
        max_runup_percent (Decimal): Maximum unrealized profit as a percentage.
        max_drawdown (Decimal): Maximum unrealized loss during the trade.
        max_drawdown_percent (Decimal): Maximum unrealized loss as a percentage.
    """

    pnl: Decimal = Decimal("0")
    pnl_percent: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    max_runup: Decimal = Decimal("0")
    max_runup_percent: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_percent: Decimal = Decimal("0")


class Trade:
    """
    Represents a trading position, including entry and exit information, and various metrics.

    This class manages the lifecycle of a trade, from entry to exit, and calculates
    various performance metrics throughout the trade's duration.
    """

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
        commission_rate: Optional[Decimal] = None,
        strategy_id: Optional[str] = None,
    ):
        """
        Initialize a new Trade instance.

        Args:
            trade_id (int): Unique identifier for the trade.
            entry_order (Order): The order that opened this trade.
            entry_bar (Bar): The price bar at which the trade was entered.
            commission_rate (Optional[Decimal]): The commission rate for the trade, if applicable.
            strategy_id (Optional[str]): The ID of the strategy that created this trade.
        """
        self.id: int = trade_id
        self.ticker: str = entry_order.details.ticker
        self.direction: Order.Direction = entry_order.details.direction
        self.initial_size: Decimal = entry_order.details.size
        self.current_size: Decimal = self.initial_size

        self.entry_price: Decimal = entry_order.fill_price or entry_order.details.price
        self.entry_timestamp: datetime = (
            entry_order.fill_timestamp or entry_bar.timestamp
        )
        self.entry_bar: Bar = entry_bar

        self.exit_price: Optional[Decimal] = None
        self.exit_timestamp: Optional[datetime] = None
        self.exit_bar: Optional[Bar] = None

        self.parent_id: Optional[int] = entry_order.details.parent_id
        self.status: Trade.Status = Trade.Status.ACTIVE
        self.metrics: TradeMetrics = TradeMetrics()
        self.alpha_name: Optional[str] = getattr(
            entry_order.details, "alpha_name", None
        )

        self.entry_order: Order = entry_order
        self.exit_orders: List[Order] = []

        self.commission_rate: Optional[Decimal] = commission_rate
        self.strategy_id: Optional[str] = strategy_id or entry_order.details.strategy_id
        self._calculate_entry_commission_and_slippage()

    def _calculate_entry_commission_and_slippage(self):
        """
        Calculates and sets the commission and slippage for the entry order.
        """
        if self.commission_rate:
            self.metrics.commission = (
                self.entry_price * self.initial_size * self.commission_rate
            )

        if self.entry_order.details.slippage:
            self.metrics.slippage = (
                abs(self.entry_order.details.price - self.entry_price)
                * self.initial_size
            )

    def update(self, current_bar: Bar) -> None:
        """
        Updates the trade metrics based on the current market price.

        This method recalculates the unrealized P&L and updates the max runup and drawdown.

        Args:
            current_bar (Bar): The current price bar representing the latest market data.
        """
        current_price = current_bar.close
        unrealized_pnl = self._calculate_pnl(current_price)
        unrealized_pnl_percent = self._calculate_pnl_percent(unrealized_pnl)

        self.metrics.pnl = unrealized_pnl
        self.metrics.pnl_percent = unrealized_pnl_percent

        self._update_runup_drawdown(unrealized_pnl, unrealized_pnl_percent)

    def close(self, exit_order: Order, exit_bar: Bar) -> None:
        """
        Closes the trade (fully or partially) with the given exit order and updates relevant information.

        This method finalizes the trade, setting its status to closed or partially closed and calculating final metrics.

        Args:
            exit_order (Order): The order used to close the trade.
            exit_bar (Bar): The price bar at which the trade was closed.
        """
        if exit_order.details.size > self.current_size:
            logger_main.log_and_raise(
                ValueError(
                    f"Exit order size ({exit_order.details.size}) exceeds current trade size ({self.current_size})"
                )
            )

        self.exit_orders.append(exit_order)
        exit_price = exit_order.fill_price or exit_order.details.price
        exit_size = exit_order.details.size

        # Update the current size
        self.current_size -= exit_size

        # Calculate P&L for this exit
        exit_pnl = self._calculate_pnl(exit_price, exit_size)
        exit_pnl_percent = self._calculate_pnl_percent(exit_pnl, exit_size)

        # Update overall metrics
        self.metrics.pnl += exit_pnl

        # Update the overall PNL percent based on the weighted average of previous and current exit
        if self.initial_size == exit_size:
            self.metrics.pnl_percent = exit_pnl_percent
        else:
            previous_weight = (self.initial_size - exit_size) / self.initial_size
            current_weight = exit_size / self.initial_size
            self.metrics.pnl_percent = (
                self.metrics.pnl_percent * previous_weight
                + exit_pnl_percent * current_weight
            )
        # Calculate and add commission and slippage for this exit
        if self.commission_rate:
            exit_commission = exit_price * exit_size * self.commission_rate
            self.metrics.commission += exit_commission

        if exit_order.details.slippage:
            exit_slippage = abs(exit_order.details.price - exit_price) * exit_size
            self.metrics.slippage += exit_slippage

        # Update trade status
        if self.current_size == 0:
            self.status = Trade.Status.CLOSED
            self.exit_price = exit_price
            self.exit_timestamp = exit_order.fill_timestamp or exit_bar.timestamp
            self.exit_bar = exit_bar
        else:
            self.status = Trade.Status.PARTIALLY_CLOSED

        self._finalize_metrics()

    def _calculate_pnl(
        self, current_price: Decimal, size: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculates the profit/loss for the trade.

        Args:
            current_price (Decimal): The current market price.
            size (Optional[Decimal]): The size to calculate P&L for. If None, uses current_size.

        Returns:
            Decimal: The calculated P&L.
        """
        size = size or self.current_size
        price_diff = (current_price - self.entry_price) * self.direction.value
        return price_diff * size

    def _calculate_pnl_percent(
        self, pnl: Decimal, size: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculates the profit/loss percentage for the trade.

        Args:
            pnl (Decimal): The calculated P&L.
            size (Optional[Decimal]): The size to calculate P&L percent for. If None, uses current_size.

        Returns:
            Decimal: The P&L as a percentage.
        """
        size = size or self.current_size
        return (pnl / (self.entry_price * size)) * Decimal("100")

    def _update_runup_drawdown(self, pnl: Decimal, pnl_percent: Decimal) -> None:
        """
        Updates the maximum runup and drawdown for the trade.

        Args:
            pnl (Decimal): The current P&L.
            pnl_percent (Decimal): The current P&L as a percentage.
        """
        if pnl > self.metrics.max_runup:
            self.metrics.max_runup = pnl
            self.metrics.max_runup_percent = pnl_percent
        elif pnl < self.metrics.max_drawdown:
            self.metrics.max_drawdown = pnl
            self.metrics.max_drawdown_percent = pnl_percent

    def _finalize_metrics(self) -> None:
        """
        Finalizes the trade metrics upon closing or partial closing.
        """
        # The metrics are continuously updated in the close method,
        # so we don't need to do much here unless there are any final calculations.
        pass

    def duration(self) -> datetime.timedelta:
        """
        Calculates the duration of the trade.

        Returns:
            datetime.timedelta: The time elapsed between trade entry and exit (or current time if still open).
        """
        end_time = self.exit_timestamp or datetime.now()
        return end_time - self.entry_timestamp

    def __repr__(self) -> str:
        """
        Returns a string representation of the Trade object.

        Returns:
            str: A concise string representation of the trade.
        """
        return (
            f"Trade(id={self.id}, ticker={self.ticker}, direction={self.direction.name}, "
            f"initial_size={self.initial_size}, current_size={self.current_size}, "
            f"entry_price={self.entry_price}, status={self.status.name}, "
            f"pnl={self.metrics.pnl:.2f}, pnl_percent={self.metrics.pnl_percent:.2f}%)"
        )

    def to_dict(self) -> dict:
        """
        Converts the Trade object to a dictionary for serialization.

        Returns:
            dict: A dictionary representation of the Trade object.
        """
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
        """
        Creates a Trade object from a dictionary representation.

        Args:
            data (dict): A dictionary containing trade data.

        Returns:
            Trade: A new Trade instance created from the provided data.
        """
        entry_order = Order(
            order_id=data["id"],
            details=OrderDetails(
                ticker=data["ticker"],
                direction=Order.Direction[data["direction"]],
                size=Decimal(data["initial_size"]),
                price=Decimal(data["entry_price"]),
                exectype=Order.ExecType.MARKET,  # Assuming market order for simplicity
                timestamp=datetime.fromisoformat(data["entry_timestamp"]),
                timeframe=Timeframe("1m"),  # Assuming 1-minute timeframe for simplicity
                strategy_id=data.get("strategy_id"),
            ),
        )
        entry_bar = Bar(
            open=Decimal(data["entry_price"]),
            high=Decimal(data["entry_price"]),
            low=Decimal(data["entry_price"]),
            close=Decimal(data["entry_price"]),
            volume=0,
            timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            timeframe=Timeframe("1m"),
            ticker=data["ticker"],
        )
        trade = cls(
            data["id"], entry_order, entry_bar, strategy_id=data.get("strategy_id")
        )
        trade.current_size = Decimal(data["current_size"])
        trade.metrics = TradeMetrics(
            pnl=Decimal(data["pnl"]),
            pnl_percent=Decimal(data["pnl_percent"]),
            commission=Decimal(data["commission"]),
            slippage=Decimal(data["slippage"]),
            max_runup=Decimal(data["max_runup"]),
            max_runup_percent=Decimal(data["max_runup_percent"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            max_drawdown_percent=Decimal(data["max_drawdown_percent"]),
        )
        trade.status = Trade.Status[data["status"]]
        trade.alpha_name = data["alpha_name"]
        if data["exit_price"]:
            trade.exit_price = Decimal(data["exit_price"])
            trade.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"])
        return trade
