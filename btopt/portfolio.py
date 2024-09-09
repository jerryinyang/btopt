import uuid
from copy import copy, deepcopy
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .data.bar import Bar
from .data.timeframe import Timeframe
from .order import (
    BracketGroup,
    BracketOrderDetails,
    OCAGroup,
    OCAOrderDetails,
    OCOGroup,
    OCOOrderDetails,
    Order,
    OrderDetails,
)
from .portfolio_managers import (
    AccountManager,
    OrderManager,
    Position,
    PositionManager,
    RiskManager,
    TradeManager,
)
from .trade import Trade
from .types import EngineType
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import logger_main


class Portfolio:
    """
    A comprehensive portfolio management class that coordinates between various specialized managers.

    This class is responsible for delegating tasks to specialized manager classes and maintaining overall portfolio state.
    """

    def __init__(
        self,
        engine: EngineType,
        initial_capital: ExtendedDecimal,
        commission_rate: ExtendedDecimal,
        margin_ratio: ExtendedDecimal,
        risk_manager_config: Dict[str, Any],
    ):
        """
        Initialize the Portfolio with its component managers.

        Args:
            engine: The trading engine instance.
            initial_capital: The starting capital for the portfolio.
            commission_rate: The commission rate for trades.
            margin_ratio: The margin ratio for leverage.
            risk_manager_config: Configuration parameters for the RiskManager.
        """
        self.engine = engine
        self.order_manager = OrderManager()
        self.trade_manager = TradeManager(commission_rate)
        self.position_manager = PositionManager()
        self.account_manager = AccountManager(initial_capital, margin_ratio)
        self.risk_manager = RiskManager(
            symbols=engine._dataview.symbols,
            **risk_manager_config,
        )

        self.metrics = pd.DataFrame(
            columns=[
                "timestamp",
                "cash",
                "equity",
                "asset_value",
                "liabilities",
                "open_pnl",
                "closed_pnl",
                "portfolio_return",
            ]
        )

    # region Update Methods

    def update(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the portfolio state based on the current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """

        self._process_pending_orders(timestamp, market_data)
        self._update_open_trades(market_data)
        self._update_positions_and_account(market_data)
        self._update_metrics(timestamp, market_data)
        self._check_and_handle_margin_call(market_data)

    def _close_trades(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> None:
        """
        Close trades based on the given order.

        Args:
            order (Order): The order that triggers trade closure.
            execution_price (ExtendedDecimal): The execution price for closing trades.
            bar (Bar): The current price bar.
        """
        remaining_size = order.get_filled_size()
        symbol = order.details.ticker
        for trade in self.trade_manager.get_trades_for_symbol(symbol):
            if remaining_size <= ExtendedDecimal("0"):
                break

            if trade.close(order, execution_price, bar, remaining_size):
                remaining_size -= trade.current_size

        if remaining_size > ExtendedDecimal("0"):
            logger_main.warning(
                f"Unable to fully close position for {symbol}. Remaining size: {remaining_size}"
            )

    def _update_positions_and_account(
        self, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update positions and account state based on current market data.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }

        long_value = self.position_manager.get_long_position_value(current_prices)
        short_value = self.position_manager.get_short_position_value(current_prices)
        unrealized_pnl = self.position_manager.get_total_unrealized_pnl(current_prices)

        self.account_manager.update_unrealized_pnl(unrealized_pnl)
        self.account_manager.update_margin_used(long_value, short_value)

    def _update_metrics(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update portfolio metrics.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """

        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }

        equity = self.account_manager.equity

        asset_value = self.position_manager.get_long_position_value(current_prices)
        liabilities = self.position_manager.get_short_position_value(current_prices)
        open_pnl = self.position_manager.get_total_unrealized_pnl(current_prices)
        closed_pnl = self.account_manager.realized_pnl
        portfolio_return = self._calculate_return(equity)

        new_metrics = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "cash": [self.account_manager.cash],
                "equity": [equity],
                "asset_value": [asset_value],
                "liabilities": [liabilities],
                "open_pnl": [open_pnl],
                "closed_pnl": [closed_pnl],
                "portfolio_return": [portfolio_return],
            }
        )
        if self.metrics.empty:
            self.metrics = new_metrics

        self.metrics = pd.concat([self.metrics, new_metrics], ignore_index=True)

    def _calculate_return(self, current_equity: ExtendedDecimal) -> float:
        """
        Calculate the portfolio return based on the current equity.

        Args:
            current_equity (ExtendedDecimal): The current portfolio equity.

        Returns:
            float: The calculated portfolio return.
        """
        if len(self.metrics) > 0:
            previous_equity = self.metrics.iloc[-1]["equity"]
            return float((current_equity - previous_equity) / previous_equity)
        return 0.0

    # endregion

    # region Order Management

    def create_order(
        self, order_details: OrderDetails, activated: bool = True
    ) -> Optional[Order]:
        """
        Create a new order and add it to the order manager.

        Args:
            order_details (OrderDetails): The details of the order to be created.
            activated (bool): Controls if the order should be activated or deactivated

        Returns:
            Optional[Order]: The created Order object, or None if margin requirements are not met.
        """
        if self._check_margin_requirements(order_details):
            return self.order_manager.create_order(order_details, activated=activated)

        else:
            logger_main.warning(f"Insufficient margin to create order: {order_details}")
            return None

    def create_oco_order(
        self, oco_details: OCOOrderDetails
    ) -> Tuple[Order, Order, OCOGroup]:
        """
        Create a One-Cancels-the-Other (OCO) order.

        Args:
            oco_details: An OCOOrderDetails object containing the details for both orders.

        Returns:
            A tuple containing both orders and the OCOGroup.
        """
        limit_order = self.create_order(oco_details.limit_order)
        stop_order = self.create_order(oco_details.stop_order)
        oco_group = self.order_manager.create_oco_group(limit_order, stop_order)
        return limit_order, stop_order, oco_group

    def create_oca_order(
        self, oca_details: OCAOrderDetails
    ) -> Tuple[List[Order], OCAGroup]:
        """
        Create a One-Cancels-All (OCA) order.

        Args:
            oca_details: An OCAOrderDetails object containing the details for all orders.

        Returns:
            A tuple containing a list of all orders and the OCAGroup.
        """
        orders = [self.create_order(details) for details in oca_details.orders]
        oca_group = self.order_manager.create_oca_group(orders)
        return orders, oca_group

    def create_bracket_order(
        self, bracket_details: BracketOrderDetails
    ) -> Tuple[Order, Optional[Order], Optional[Order], BracketGroup]:
        """
        Create a Bracket order.

        Args:
            bracket_details (BracketOrderDetails): A BracketOrderDetails object containing the details for entry,
                                                optional take profit, and optional stop loss orders.

        Returns:
            Tuple[Order, Optional[Order], Optional[Order], BracketGroup]: A tuple containing the entry order,
            take profit order (if specified), stop loss order (if specified), and the BracketGroup.
        """
        entry_order = self.create_order(bracket_details.entry_order)
        take_profit_order = (
            self.create_order(bracket_details.take_profit_order, activated=False)
            if bracket_details.take_profit_order
            else None
        )
        stop_loss_order = (
            self.create_order(bracket_details.stop_loss_order, activated=False)
            if bracket_details.stop_loss_order
            else None
        )

        bracket_group = self.order_manager.create_bracket_order(
            entry_order, take_profit_order, stop_loss_order
        )

        logger_main.warning(
            f"\n----- CREATED BRACKET ORDER -----\nENTRY: {bracket_group.entry_order.is_active}\n"
            + f"LIMIT: {bracket_group.take_profit_order.is_active}\nSTOP: {bracket_group.stop_loss_order.is_active}\n"
            + f"PENDING ORDERS: {''.join(f'{order} | {order.is_active}\n' for order in self.order_manager.orders.values())}\n\n"
        )
        return entry_order, take_profit_order, stop_loss_order, bracket_group

    def create_complex_order(
        self,
        order_details: Union[OCOOrderDetails, OCAOrderDetails, BracketOrderDetails],
    ) -> Union[
        Tuple[Order, Order, OCOGroup],
        Tuple[List[Order], OCAGroup],
        Tuple[Order, Optional[Order], Optional[Order], BracketGroup],
    ]:
        """
        Create a complex order (OCO, OCA, or Bracket) and add it to the order manager.

        Args:
            order_details: Either OCOOrderDetails, OCAOrderDetails, or BracketOrderDetails.

        Returns:
            A tuple containing the created orders and the corresponding order group.

        Raises:
            ValueError: If an invalid order_details type is provided.
        """
        if isinstance(order_details, OCOOrderDetails):
            return self.create_oco_order(order_details)
        elif isinstance(order_details, OCAOrderDetails):
            return self.create_oca_order(order_details)
        elif isinstance(order_details, BracketOrderDetails):
            return self.create_bracket_order(order_details)
        else:
            logger_main.log_and_raise(
                ValueError(f"Invalid order details type: {type(order_details)}")
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if the order was successfully cancelled, False otherwise.
        """
        return self.order_manager.cancel_order_by_id(order_id)

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Process all pending orders based on current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        # Process the pending orders; get the executed orders
        filled_orders = self.order_manager.process_orders(timestamp, market_data)
        for order in filled_orders:
            self._execute_order(order, market_data)

    def _execute_order(
        self,
        order: Order,
        market_data: Dict[str, Dict[Timeframe, Bar]],
        recursive: bool = False,
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update all relevant components.

        Args:
            order (Order): The order to execute.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing a boolean indicating if the order was executed
            and the resulting Trade object if applicable.
        """
        bar: Bar = market_data[order.details.ticker][order.details.timeframe]
        symbol = order.details.ticker
        direction = order.details.direction
        execution_price = order.get_last_fill_price() or bar.close
        fill_size = order.get_last_fill_size()

        entry_order = deepcopy(order)
        exit_order = deepcopy(order)

        # Check risk limits / margin requirements
        is_position_expanding, is_position_reversing = self._check_if_new_trade(
            symbol, direction, fill_size
        )

        # Update the position
        old_position = self.position_manager.get_position(symbol).copy()

        # Execute the orders
        logger_main.warning(
            f"\n\n\nCURRENT QUANTITY: {old_position.quantity}\nNEW QUANTITY: {fill_size * direction.value}\nIS EXPANDING: {is_position_expanding}\nIS REVERSING: {is_position_reversing}"
        )

        if not recursive:
            # Split the order into entry and exit sizes
            exit_size = abs(old_position.quantity * -1)  # Reverse the current position
            entry_size = abs((fill_size * direction.value) + old_position.quantity)

            if is_position_reversing:
                # Split the order into two orders; exit and entry order; set their sizes
                entry_order.set_last_fill_size(entry_size)
                exit_order.set_last_fill_size(exit_size)

                # Execute the orders
                logger_main.warning(
                    f"\n\nATTEMPTING REVERSE SEQUENCE!!!\nEXIT SIZE: {exit_size}\nENTRY SIZE: {entry_size}\n"
                )
                self._execute_order(exit_order, market_data, recursive=True)
                return self._execute_order(entry_order, market_data, recursive=True)

        if is_position_expanding:
            unrealized_pnl = self.trade_manager.calculate_unrealized_pnl(
                symbol, execution_price
            )

            # Check risk limits
            if not self.risk_manager.check_risk_limits(
                order,
                fill_size,
                self.account_manager.equity + unrealized_pnl,
                self.position_manager.get_all_positions(),
            ):
                self._reject_order(order, "Risk limits breached")
                return False, None

        # Update the position
        self.position_manager.update_position(order, execution_price, fill_size)
        new_position = self.position_manager.get_position(symbol)

        # Calculate realized PnL from position
        realized_pnl = self._calculate_realized_pnl(
            old_position, new_position, execution_price
        )

        # Update trade
        trade = self.trade_manager.manage_trade(
            order,
            bar,
            old_position,
        )
        logger_main.warning(
            f"\n\nOPEN TRADES: {self.trade_manager.get_open_trades()}\n\n"
        )

        # Update account
        self._update_account_on_execution(order, execution_price, realized_pnl)
        logger_main.info(
            f"Executed order: {order.id}, Symbol: {symbol}, "
            f"Price: {execution_price}, Size: {fill_size}"
        )

        return True, trade

    def _reject_order(self, order: Order, reason: str) -> None:
        """
        Reject an order and update its status.

        Args:
            order (Order): The order to be rejected.
            reason (str): The reason for rejection.
        """
        order.status = Order.Status.REJECTED
        self.order_manager.updated_orders.append(order)
        logger_main.warning(f"Order {order.id} rejected: {reason}")

    def _calculate_realized_pnl(
        self,
        old_position: Position,
        new_position: Position,
        execution_price: ExtendedDecimal,
    ) -> ExtendedDecimal:
        """
        Calculate the realized PnL from a position change.

        Args:
            old_position (Position): The position before the order execution.
            new_position (Position): The position after the order execution.
            execution_price (ExtendedDecimal): The price at which the order was executed.

        Returns:
            ExtendedDecimal: The realized PnL.
        """
        if old_position.quantity == 0:
            # No prior existing positions
            return ExtendedDecimal("0")

        closed_quantity = abs(old_position.quantity) - abs(new_position.quantity)
        if closed_quantity <= 0:
            # New order increased the position (same direction)
            return ExtendedDecimal("0")

        # New order closed a portion or all of the position
        avg_entry_price = old_position.average_price
        realized_pnl = (
            (execution_price - avg_entry_price)
            * closed_quantity
            * (1 if old_position.quantity > 0 else -1)
        )
        return realized_pnl

    def _check_if_new_trade(
        self, symbol: str, direction: Order.Direction, fill_size: ExtendedDecimal
    ):
        """Returns true if the new order is leading to a new trade"""

        # Check for position expansion or reversal
        old_position = self.position_manager.get_position(symbol)
        old_quantity = old_position.quantity
        new_quantity = fill_size * direction.value

        is_position_expanding = (
            (old_quantity == 0)
            or (old_quantity > 0 and (old_quantity + new_quantity > old_quantity))
            or (old_quantity < 0 and (old_quantity + new_quantity < old_quantity))
        )
        is_position_reversing = (
            old_quantity > 0 and (old_quantity + new_quantity < 0)
        ) or (old_quantity < 0 and (old_quantity + new_quantity > 0))

        return (is_position_expanding, is_position_reversing)

    # endregion

    # region Position Management

    def close_all_positions(self, timestamp: datetime) -> None:
        """
        Close all open positions in the portfolio.

        Args:
            timestamp (datetime): The current timestamp for order creation and processing.
        """
        for symbol, position in self.position_manager.positions.items():
            if position != ExtendedDecimal("0"):
                close_order = self._create_market_order_to_close(
                    symbol, abs(position.quantity)
                )
                self._execute_order(close_order, self.engine.get_current_data(symbol))

        logger_main.info("All positions have been closed.")

    def _create_market_data_for_order(
        self, order: Order
    ) -> Dict[str, Dict[Timeframe, Bar]]:
        """
        Create a market data structure for a single order, used for order processing.

        Args:
            order (Order): The order for which to create market data.

        Returns:
            Dict[str, Dict[Timeframe, Bar]]: A market data structure for the order.
        """
        current_price = self._get_current_prices()[order.details.ticker]
        dummy_bar = Bar(
            open=current_price,
            high=current_price,
            low=current_price,
            close=current_price,
            volume=0,
            timestamp=order.details.timestamp,
            timeframe=order.details.timeframe or self.engine.default_timeframe,
            ticker=order.details.ticker,
        )
        return {order.details.ticker: {dummy_bar.timeframe: dummy_bar}}

    def get_position_size(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            ExtendedDecimal: The current position size (positive for long, negative for short, 0 for no position).
        """
        position = self.position_manager.get_position(symbol)
        return position.quantity

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dict[str, Position]: A dictionary mapping symbols to their current Position objects.
        """
        return self.position_manager.get_all_positions()

    # endregion

    # region Account Management
    def _update_account_on_execution(
        self,
        order: Order,
        execution_price: ExtendedDecimal,
        realized_pnl: ExtendedDecimal,
    ) -> None:
        """
        Update the account after an order execution.

        This method updates the account's cash balance based on the realized PnL and commissions.
        It does not include the full position value in cash updates.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            realized_pnl (ExtendedDecimal): The realized PnL from this order execution.
        """
        commission = (
            execution_price * order.details.size * self.trade_manager.commission_rate
        )
        cash_change = realized_pnl - commission
        self.account_manager.update_cash(
            cash_change,
            f"Order execution for {order.details.ticker}: PnL {realized_pnl}, Commission {commission}",
        )

    def get_account_value(self) -> ExtendedDecimal:
        """
        Get the current total account value (equity).

        Returns:
            The current account value.
        """
        return self.account_manager.equity

    def get_available_margin(self) -> ExtendedDecimal:
        """
        Get the available margin for new trades.

        Returns:
            The available margin.
        """
        return self.account_manager.buying_power

    # endregion

    # region Trade Management
    def get_closed_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        """
        Get all closed trades, optionally filtered by strategy ID.

        Args:
            strategy_id: The ID of the strategy to filter trades for.

        Returns:
            A list of closed trades.
        """
        return self.trade_manager.get_closed_trades(strategy_id)

    def _update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Update all open trades based on current market data.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        self.trade_manager.update_open_trades(market_data)

    # endregion

    # region Portfolio State and Reporting

    def get_portfolio_state(
        self, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state,
            including detailed position information.
        """
        positions_info = {
            symbol: {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "last_update_time": position.last_update_time,
                "cost_basis": position.total_cost,
            }
            for symbol, position in self.position_manager.get_all_positions().items()
        }
        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }

        return {
            "cash": self.account_manager.cash,
            "equity": self.account_manager.equity,
            "margin_used": self.account_manager.margin_used,
            "buying_power": self.account_manager.get_buying_power(),
            "realized_pnl": self.account_manager.realized_pnl,
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(
                current_prices
            ),
            "positions": positions_info,
            "open_trades": self.trade_manager.get_open_trades(),
            "pending_orders": self.order_manager.get_pending_orders(),
        }

    def get_metrics_data(self) -> pd.DataFrame:
        """
        Retrieve the complete metrics DataFrame.

        Returns:
            The complete metrics DataFrame.
        """
        return self.metrics

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete trade history.

        Returns:
            A list of dictionaries, each representing a trade.
        """
        return pd.DataFrame(
            trade.to_dict() for trade in self.trade_manager.get_closed_trades()
        )

    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete order history.

        Returns:
            A list of dictionaries, each representing an order.
        """
        return pd.DataFrame(
            order.to_dict() for order in self.order_manager.get_all_orders()
        )

    # endregion

    # region Risk Management
    def calculate_risk_amount(self, symbol: str) -> ExtendedDecimal:
        """Calculate the risk amount for a specific symbol."""
        return self.risk_manager.calculate_risk_amount(
            symbol, self.account_manager.equity
        )

    def _check_and_handle_margin_call(
        self, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Check for margin call and handle it if necessary.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        if self.risk_manager.check_margin_call(
            self.account_manager.equity,
            self.account_manager.margin_used,
            self.account_manager.margin_ratio,
        ):
            self._handle_margin_call(market_data)

    def _check_margin_requirements(self, order_details: OrderDetails) -> bool:
        """
        Check if there's sufficient margin to execute the order, using the RiskManager.

        This method delegates the margin and risk checks to the RiskManager,
        which provides a comprehensive assessment of the order's viability.

        Args:
            order_details (OrderDetails): The details of the order to check.

        Returns:
            bool: True if the order passes margin and risk checks, False otherwise.
        """

        details = copy(order_details)
        order = Order(str(uuid.uuid4()), details)

        # Check if the order is a reversal
        _, is_reversal = self._check_if_new_trade(
            details.ticker,
            details.direction,
            details.size,
        )

        if is_reversal:
            details = replace(
                details,
                size=abs(
                    (details.size * details.direction.value)
                    + self.position_manager.get_position(details.ticker).quantity
                ),
            )

        logger_main.warning(
            f"\n\nORIGINAL SIZE: {order_details.size}\nFINAL SIZE: {details.size}\n\n"
        )

        order = Order(str(uuid.uuid4()), details)  # Create a temporary Order object

        account_value = self.account_manager.equity
        current_positions = self.position_manager.get_all_positions()

        if self.risk_manager.check_margin_requirements(
            order, account_value, current_positions
        ):
            logger_main.info(f"Margin and risk checks passed for order: {details}")
            return True
        else:
            logger_main.warning(f"Order failed margin or risk checks: {details}")
            return False

    def _handle_margin_call(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Handle a margin call by reducing positions until margin requirements are met.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }
        positions_to_reduce = self.risk_manager.handle_margin_call(
            self.position_manager.positions,
            current_prices,
            self.trade_manager.get_open_trades(),
            self.account_manager.equity,
            self.account_manager.margin_used,
            self.account_manager.margin_ratio,
        )

        for symbol, reduce_amount in positions_to_reduce:
            close_order = self._create_market_order_to_close(symbol, reduce_amount)
            self._execute_order(
                close_order, market_data[symbol][min(market_data[symbol].keys())]
            )

        logger_main.info("Margin call handled. Positions have been reduced.")

    def _create_market_order_to_close(
        self, symbol: str, size: ExtendedDecimal
    ) -> Order:
        """
        Create a market order to close a position.

        Args:
            symbol (str): The symbol of the position to close.
            size (ExtendedDecimal): The size of the position to close.

        Returns:
            Order: A market order to close the position.
        """
        current_position = self.position_manager.get_position(symbol)
        direction = (
            Order.Direction.SHORT
            if current_position.quantity > 0
            else Order.Direction.LONG
        )
        bar = self.engine.get_current_data(symbol)
        order_details = OrderDetails(
            ticker=symbol,
            direction=direction,
            size=abs(size),
            price=bar.close,
            exectype=Order.ExecType.MARKET,
            timestamp=self.engine._current_timestamp,
            timeframe=self.engine.default_timeframe,
            strategy_id="CLOSE_POSITION",
        )
        return self.create_order(order_details)

    def update_risk_parameters(self, **kwargs: Any) -> None:
        """
        Update risk management parameters.

        Args:
            **kwargs: Risk management parameters to update.
        """
        self.risk_manager.update_parameters(**kwargs)

    def _notify_margin_call_status(self) -> None:
        """
        Notify about the margin call status.
        """
        # Implementation depends on your notification system
        logger_main.warning("Margin call occurred and has been handled.")

    # endregion

    # region Symbol Weight Management
    def set_symbol_weight(self, symbol: str, weight: float) -> None:
        """Set the weight for a specific symbol."""
        self.risk_manager.set_symbol_weights({symbol: ExtendedDecimal(str(weight))})

    def get_symbol_weight(self, symbol: str) -> float:
        """Get the weight of a specific symbol."""
        return float(self.risk_manager.get_symbol_weight(symbol))

    def set_all_symbol_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for all symbols."""
        extended_weights = {
            symbol: ExtendedDecimal(str(weight)) for symbol, weight in weights.items()
        }
        self.risk_manager.set_all_symbol_weights(extended_weights)

    def get_all_symbol_weights(self) -> Dict[str, float]:
        """Get weights for all symbols."""
        return {
            symbol: float(weight)
            for symbol, weight in self.risk_manager.get_all_symbol_weights().items()
        }

    # endregion

    # region Utility Methods

    def _get_current_prices(self) -> Dict[str, ExtendedDecimal]:
        """
        Get the current prices for all symbols in the portfolio.

        Returns:
            A dictionary mapping symbols to their current prices.
        """
        return {
            symbol: self.engine.get_current_data(symbol).close
            for symbol in self.engine._dataview.symbols
        }

    def handle_order_group_events(self) -> None:
        """
        Handle events from order groups (e.g., when an OCO order is filled).
        """
        self.order_manager.handle_order_group_events()

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.
        """
        self.__init__(
            self.engine,
            self.account_manager.initial_capital,
            self.trade_manager.commission_rate,
            self.account_manager.margin_ratio,
            {
                attr: getattr(self.risk_manager, attr)
                for attr in self.risk_manager.__dict__
                if not attr.startswith("_")
            },
        )
        logger_main.info("Portfolio reset to initial state.")

    # endregion
