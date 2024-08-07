from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main
from ..order import Order
from ..parameters import Parameters
from ..trade import Trade
from .helper_new import Data


class StrategyError(Exception):
    pass


class Strategy(ABC):
    def __init__(
        self,
        name: str,
        symbol: str,
        timeframe: Union[str, Timeframe],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name: str = name
        self.symbol: str = symbol
        self.timeframe: Timeframe = (
            Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        )
        self._parameters: Parameters = Parameters(parameters or {})
        self.data: Data = Data(symbol, self.timeframe)
        self._engine: Any = None
        self._initialized: bool = False
        self._positions: Dict[str, Decimal] = {}
        self._pending_orders: List[Order] = []
        self._open_trades: Dict[str, List[Trade]] = {}
        self._closed_trades: List[Trade] = []

    # region Initialization and Configuration

    def initialize(self) -> None:
        if self._initialized:
            logger_main.warning(f"Strategy {self.name} is already initialized.")
            return
        self._initialized = True
        logger_main.info(f"Initialized strategy: {self.name}")

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters: Dict[str, Any]) -> None:
        self._parameters = Parameters(new_parameters)
        logger_main.info(
            f"Updated parameters for strategy {self.name}: {new_parameters}"
        )

    # endregion

    # region Data Access

    def get_data(
        self,
        value: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[Union[Timeframe, str]] = None,
        size: int = 1,
    ) -> Any:
        return self.data.get(value, symbol, timeframe, size)

    # endregion

    # region Order Management

    def buy(
        self, size: Decimal, price: Optional[Decimal] = None, **kwargs: Any
    ) -> Optional[Order]:
        if self._engine is None:
            logger_main.error("Strategy is not connected to an engine.")
            raise StrategyError("Strategy is not connected to an engine.")
        order = self._engine.create_order(
            self.symbol, Order.Direction.LONG, size, price, **kwargs
        )
        if order:
            self._pending_orders.append(order)
        return order

    def sell(
        self, size: Decimal, price: Optional[Decimal] = None, **kwargs: Any
    ) -> Optional[Order]:
        if self._engine is None:
            logger_main.error("Strategy is not connected to an engine.")
            raise StrategyError("Strategy is not connected to an engine.")
        order = self._engine.create_order(
            self.symbol, Order.Direction.SHORT, size, price, **kwargs
        )
        if order:
            self._pending_orders.append(order)
        return order

    def cancel(self, order: Order) -> bool:
        if self._engine is None:
            logger_main.error("Strategy is not connected to an engine.")
            raise StrategyError("Strategy is not connected to an engine.")
        success = self._engine.cancel_order(order)
        if success:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
        return success

    def close(self, symbol: Optional[str] = None) -> bool:
        if self._engine is None:
            logger_main.error("Strategy is not connected to an engine.")
            raise StrategyError("Strategy is not connected to an engine.")
        symbol = symbol or self.symbol
        success = self._engine.close_positions(symbol)
        if success:
            self._positions[symbol] = Decimal("0")
        return success

    # endregion

    # region Position Management

    def get_position(self, symbol: Optional[str] = None) -> Decimal:
        symbol = symbol or self.symbol
        return self._positions.get(symbol, Decimal("0"))

    def calculate_position_size(
        self, risk_percent: Decimal, stop_loss: Decimal
    ) -> Decimal:
        if self._engine is None:
            logger_main.error("Strategy is not connected to an engine.")
            raise StrategyError("Strategy is not connected to an engine.")
        account_value = self._engine.get_account_value()
        current_price = self.data.close
        risk_amount = account_value * (risk_percent / Decimal("100"))
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share == Decimal("0"):
            logger_main.error("Risk per share is zero. Cannot calculate position size.")
            raise StrategyError(
                "Risk per share is zero. Cannot calculate position size."
            )
        position_size = risk_amount / risk_per_share
        return self._engine.round_position_size(self.symbol, position_size)

    # endregion

    # region Order and Trade Update Handling

    def _on_order_update(self, order: Order) -> None:
        if order.status == Order.Status.FILLED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
            self._update_position(order)
        elif order.status == Order.Status.CANCELED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
        self.on_order_update(order)

    def _on_trade_update(self, trade: Trade) -> None:
        if trade.status == Trade.Status.CLOSED:
            self._update_position(trade)
            self._open_trades[trade.ticker] = [
                t for t in self._open_trades[trade.ticker] if t.id != trade.id
            ]
            self._closed_trades.append(trade)
        self.on_trade_update(trade)

    def _update_position(self, transaction: Union[Order, Trade]) -> None:
        symbol = transaction.ticker
        size = (
            transaction.size
            if isinstance(transaction, Trade)
            else transaction.filled_size
        )
        if transaction.direction == Order.Direction.LONG:
            self._positions[symbol] = self._positions.get(symbol, Decimal("0")) + size
        else:  # SHORT
            self._positions[symbol] = self._positions.get(symbol, Decimal("0")) - size

    # endregion

    # region Performance Tracking

    def calculate_metrics(self) -> Dict[str, Any]:
        if not self._closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit": Decimal("0"),
            }

        winning_trades = [t for t in self._closed_trades if t.pnl > 0]
        losing_trades = [t for t in self._closed_trades if t.pnl < 0]

        total_trades = len(self._closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_profit = sum(t.pnl for t in self._closed_trades)
        total_gains = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_profit": total_profit,
        }

    # endregion

    # region Abstract Methods

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        pass

    def on_order_update(self, order: Order) -> None:
        pass

    def on_trade_update(self, trade: Trade) -> None:
        pass

    # endregion

    # region Utility Methods

    def set_engine(self, engine: Any) -> None:
        self._engine = engine

    def __repr__(self) -> str:
        return f"Strategy(name={self.name}, symbol={self.symbol}, timeframe={self.timeframe})"

    # endregion
