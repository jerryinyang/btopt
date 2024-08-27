import uuid
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
    PositionManager,
    RiskManager,
    TradeManager,
)
from .reporter import Reporter
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

        self.reporter: Optional[Reporter] = None
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

    # def update(
    #     self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    # ) -> None:
    #     """
    #     Update the portfolio state based on current market data.

    #     This method coordinates updates across all manager components, processes orders,
    #     handles delayed trade creations, and updates metrics.

    #     Args:
    #         timestamp (datetime): The current timestamp.
    #         market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.
    #     """
    #     # Process orders using the order manager
    #     filled_orders: List[Order] = self.order_manager.process_orders(
    #         timestamp, market_data
    #     )

    #     # Handle filled orders
    #     for order in filled_orders:
    #         self._handle_filled_order(order, market_data)

    #     # Update positions and account
    #     self._update_positions_and_account(market_data)

    #     # Update metrics
    #     self._update_metrics(timestamp)

    #     # Check for margin call
    #     if self.risk_manager.check_margin_call(
    #         self.account_manager.equity, self.account_manager.margin_used
    #     ):
    #         self._handle_margin_call(market_data)

    def _handle_filled_order(
        self, order: Order, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Handle a filled order by updating positions, trades, and account state.

        This method processes a filled order, updates the portfolio state, and manages any associated order groups.

        Args:
            order (Order): The filled order to be handled.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.

        Raises:
            ValueError: If the order's symbol is not found in the market data.
        """
        symbol: str = order.details.ticker

        # Ensure the symbol exists in the market data
        if symbol not in market_data:
            logger_main.log_and_raise(
                ValueError(f"Symbol {symbol} not found in market data")
            )

        current_bar: Bar = market_data[symbol][order.details.timeframe]
        execution_price: ExtendedDecimal = (
            order.get_last_fill_price() or current_bar.close
        )

        # Update position
        self.position_manager.update_position(order, execution_price)

        # Calculate and update account for the trade
        filled_size: ExtendedDecimal = order.get_filled_size()
        cost: ExtendedDecimal = execution_price * filled_size
        commission: ExtendedDecimal = cost * self.trade_manager.commission_rate
        self.account_manager.update_cash(
            -cost - commission, f"Order execution for {symbol}"
        )

        # Create or update trade
        if order.details.direction == Order.Direction.LONG:
            self.trade_manager.create_trade(order, execution_price, current_bar)
        else:
            self._close_trades(order, execution_price, current_bar)

        # Handle order group updates
        self.order_manager.handle_order_update(order)

        # Log the filled order details
        logger_main.info(
            f"Filled order: {order.id}, Symbol: {symbol}, "
            f"Price: {execution_price}, Size: {filled_size}"
        )

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
            market_data: The current market data for all symbols and timeframes.
        """
        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }

        # Update unrealized PnL for all positions
        for symbol, position in self.position_manager.get_all_positions().items():
            self.position_manager.update_unrealized_pnl(symbol, current_prices[symbol])

        # Update account equity
        total_unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        self.account_manager.update_equity(total_unrealized_pnl)

    # def _update_metrics(self, timestamp: datetime) -> None:
    #     """
    #     Update portfolio metrics.

    #     Args:
    #         timestamp: The current timestamp.
    #     """
    #     equity = self.account_manager.equity
    #     asset_value = self.position_manager.get_long_position_value(
    #         self._get_current_prices()
    #     )
    #     liabilities = self.position_manager.get_short_position_value(
    #         self._get_current_prices()
    #     )

    #     new_row = pd.DataFrame(
    #         {
    #             "timestamp": [timestamp],
    #             "cash": [self.account_manager.cash],
    #             "equity": [equity],
    #             "asset_value": [asset_value],
    #             "liabilities": [liabilities],
    #             "open_pnl": [self.position_manager.get_total_unrealized_pnl()],
    #             "closed_pnl": [self.position_manager.get_total_realized_pnl()],
    #             "portfolio_return": [self._calculate_return(equity)],
    #         }
    #     )

    #     if self.metrics.empty:
    #         self.metrics = new_row
    #     self.metrics = pd.concat([self.metrics, new_row], ignore_index=True).fillna(0)

    def _calculate_return(self, current_equity: ExtendedDecimal) -> float:
        """
        Calculate the portfolio return based on the current equity.

        Args:
            current_equity: The current portfolio equity.

        Returns:
            The calculated portfolio return.
        """
        if len(self.metrics) > 0:
            previous_equity = self.metrics.iloc[-1]["equity"]
            return float((current_equity - previous_equity) / previous_equity)
        return 0.0

    # endregion

    # region Order Management

    # def create_order(self, order_details: OrderDetails) -> Order:
    #     """
    #     Create a new order and add it to the order manager.

    #     Args:
    #         order_details (OrderDetails): The details of the order to be created.

    #     Returns:
    #         Order: The created Order object.
    #     """
    #     return self.order_manager.create_order(order_details)

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
        entry_order = self.order_manager.create_order(bracket_details.entry_order)
        take_profit_order = (
            self.order_manager.create_order(
                bracket_details.take_profit_order, activated=False
            )
            if bracket_details.take_profit_order
            else None
        )
        stop_loss_order = (
            self.order_manager.create_order(
                bracket_details.stop_loss_order, activated=False
            )
            if bracket_details.stop_loss_order
            else None
        )

        bracket_group = self.order_manager.create_bracket_order(
            entry_order, take_profit_order, stop_loss_order
        )

        logger_main.warning(
            f"\n----- CREATED BRACKET ORDER -----\nENTRY: {bracket_group.entry_order.is_active}\nLIMIT: {bracket_group.take_profit_order.is_active}\nSTOP: {bracket_group.stop_loss_order.is_active}\n\n"
        )
        return entry_order, take_profit_order, stop_loss_order, bracket_group

    # def create_complex_order(
    #     self,
    #     order_details: Union[OCOOrderDetails, OCAOrderDetails, BracketOrderDetails],
    # ) -> Union[
    #     Tuple[Order, Order, OCOGroup],
    #     Tuple[List[Order], OCAGroup],
    #     Tuple[Order, Order, Order, BracketGroup],
    # ]:
    #     """
    #     Create a complex order based on the type of order details provided.

    #     Args:
    #         order_details: Either OCOOrderDetails, OCAOrderDetails, or BracketOrderDetails.

    #     Returns:
    #         A tuple containing the created orders and the corresponding order group.

    #     Raises:
    #         ValueError: If an invalid order_details type is provided.
    #     """
    #     if isinstance(order_details, OCOOrderDetails):
    #         return self.create_oco_order(order_details)
    #     elif isinstance(order_details, OCAOrderDetails):
    #         return self.create_oca_order(order_details)
    #     elif isinstance(order_details, BracketOrderDetails):
    #         return self.create_bracket_order(order_details)
    #     else:
    #         logger_main.log_and_raise(
    #             ValueError(f"Invalid order details type: {type(order_details)}")
    #         )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if the order was successfully cancelled, False otherwise.
        """
        return self.order_manager.cancel_order(order_id)

    # endregion

    # region Position Management

    # def close_all_positions(self, timestamp: datetime) -> None:
    #     """
    #     Close all open positions in the portfolio.

    #     This method delegates the task of closing all positions to the PositionManager,
    #     and then processes the resulting orders.

    #     Args:
    #         timestamp (datetime): The current timestamp for order creation and processing.
    #     """
    #     close_orders = self.position_manager.generate_close_all_orders(
    #         timestamp,
    #         self.engine.default_timeframe,
    #         self._get_current_prices(),
    #     )

    #     for order in close_orders:
    #         self._handle_filled_order(order, self._create_market_data_for_order(order))

    #     logger_main.info("All positions have been closed.")

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
            symbol: The symbol to check.

        Returns:
            The current position size (positive for long, negative for short).
        """
        return self.position_manager.get_position(symbol)

    def get_all_positions(self) -> Dict[str, ExtendedDecimal]:
        """
        Get all current positions.

        Returns:
            A dictionary mapping symbols to their current position sizes.
        """
        return self.position_manager.get_all_positions()

    # endregion

    # region Account Management

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

    # endregion

    # region Portfolio State and Reporting

    # def get_portfolio_state(self) -> Dict[str, Any]:
    #     """
    #     Get the current state of the portfolio.

    #     Returns:
    #         A dictionary containing the current portfolio state.
    #     """
    #     return {
    #         "cash": self.account_manager.cash,
    #         "equity": self.account_manager.equity,
    #         "long_position_value": self.position_manager.get_long_position_value(
    #             self._get_current_prices()
    #         ),
    #         "short_position_value": self.position_manager.get_short_position_value(
    #             self._get_current_prices()
    #         ),
    #         "open_trades": self.trade_manager.get_open_trades(),
    #         "closed_trades_count": len(self.trade_manager.get_closed_trades()),
    #         "pending_orders": self.order_manager.get_active_orders(),
    #         "realized_pnl": self.position_manager.get_total_realized_pnl(),
    #         "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
    #         "margin_used": self.account_manager.margin_used,
    #         "buying_power": self.account_manager.buying_power,
    #         "positions": self.position_manager.get_all_positions(),
    #     }

    def set_reporter(self, reporter: Reporter) -> None:
        """
        Set the Reporter instance for this portfolio.

        Args:
            reporter: The Reporter instance to use.
        """
        self.reporter = reporter

    def generate_report(self) -> None:
        """
        Generate a performance report using the associated Reporter.
        """
        if self.reporter:
            self.reporter.generate_performance_report()
        else:
            logger_main.warning("No Reporter set. Unable to generate report.")

    def get_metrics_data(self) -> pd.DataFrame:
        """
        Retrieve the complete metrics DataFrame.

        Returns:
            The complete metrics DataFrame.
        """
        return self.metrics

    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate various performance metrics for the portfolio.

        Returns:
            A dictionary containing calculated performance metrics.
        """
        if self.reporter:
            metrics = self.reporter.calculate_performance_metrics()
            metrics.update(
                {
                    "var": self.reporter.calculate_var(),
                    "cvar": self.reporter.calculate_cvar(),
                    "current_drawdown": self.get_current_drawdown(),
                }
            )
            return metrics
        else:
            logger_main.warning(
                "No Reporter set. Unable to calculate performance metrics."
            )
            return {}

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete trade history.

        Returns:
            A list of dictionaries, each representing a trade.
        """
        return [trade.to_dict() for trade in self.trade_manager.get_closed_trades()]

    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete order history.

        Returns:
            A list of dictionaries, each representing an order.
        """
        return [order.to_dict() for order in self.order_manager.get_all_orders()]

    def get_current_drawdown(self) -> float:
        """
        Calculate the current drawdown of the portfolio.

        Returns:
            The current drawdown as a percentage.
        """
        if self.reporter:
            return self.reporter.get_current_drawdown()
        else:
            logger_main.warning(
                "No Reporter set. Unable to calculate current drawdown."
            )
            return 0.0

    def get_drawdown_details(self) -> pd.DataFrame:
        """
        Get detailed information about drawdowns.

        Returns:
            pd.DataFrame: A DataFrame containing drawdown details.
        """
        if self.reporter:
            return self.reporter.get_drawdown_details()
        else:
            logger_main.warning("No Reporter set. Unable to get drawdown details.")
            return pd.DataFrame()

    # endregion

    # region Risk Management
    def calculate_risk_amount(self, symbol: str) -> ExtendedDecimal:
        """Calculate the risk amount for a specific symbol."""
        return self.risk_manager.calculate_risk_amount(
            symbol, self.account_manager.equity
        )

    def _handle_margin_call(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Handle a margin call by gradually reducing positions until the margin requirement is met.
        """
        logger_main.warning("Margin call triggered. Attempting to resolve...")

        while self.risk_manager.check_margin_call(
            self.account_manager.equity, self.account_manager.margin_used
        ):
            current_prices = self._get_current_prices()
            position_to_reduce = self.risk_manager.select_position_to_reduce(
                self.position_manager.get_all_positions(),
                current_prices,
                self.trade_manager.get_open_trades(),
            )

            if position_to_reduce is None:
                logger_main.log_and_raise(
                    "Unable to resolve margin call. No suitable positions to reduce."
                )
                break

            symbol, reduce_size = position_to_reduce
            close_order = self._create_market_order_to_close(symbol, reduce_size)
            self._handle_filled_order(close_order, market_data)

            logger_main.info(
                f"Reduced position in {symbol} by {reduce_size} to address margin call."
            )

        if not self.risk_manager.check_margin_call(
            self.account_manager.equity, self.account_manager.margin_used
        ):
            logger_main.info("Margin call resolved successfully.")
        else:
            logger_main.warning(
                "Margin call could not be fully resolved. Account remains under margin call."
            )

        self._notify_margin_call_status()

    def _create_market_order_to_close(
        self, symbol: str, size: ExtendedDecimal
    ) -> Order:
        """Create a market order to close a position."""
        current_position = self.position_manager.get_position(symbol)
        direction = (
            Order.Direction.SHORT
            if current_position.quantity > 0
            else Order.Direction.LONG
        )
        order_details = OrderDetails(
            ticker=symbol,
            direction=direction,
            size=abs(size),
            price=None,  # Market order
            exectype=Order.ExecType.MARKET,
            timestamp=self.engine._current_timestamp,
            timeframe=self.engine.default_timeframe,
            strategy_id="MARGIN_CALL",
        )
        return self.create_order(order_details)

    def update_risk_parameters(self, **kwargs: Any) -> None:
        """
        Update risk management parameters.

        Args:
            **kwargs: Risk management parameters to update.
        """
        self.risk_manager.update_parameters(**kwargs)

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

    # region PREVIOUS MODIFICATIONS

    def _update_account_on_execution(
        self, order: Order, execution_price: ExtendedDecimal
    ) -> None:
        """
        Update the account after an order execution.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
        """
        cost = execution_price * order.details.size
        commission = cost * self.trade_manager.commission_rate
        self.account_manager.update_cash(
            -cost - commission, f"Order execution for {order.details.ticker}"
        )
        self.account_manager.update_margin(cost * self.account_manager.margin_ratio)

    def _notify_margin_call_status(self) -> None:
        """
        Notify about the margin call status.
        """
        # Implementation depends on your notification system
        logger_main.warning("Margin call occurred and has been handled.")

    # endregion

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

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Process all pending orders based on current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        executed_orders = self.order_manager.process_orders(timestamp, market_data)
        for order in executed_orders:
            self._execute_order(
                order, market_data[order.details.ticker][order.details.timeframe]
            )

    def _execute_order(self, order: Order, bar: Bar) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update all relevant components.

        Args:
            order (Order): The order to execute.
            bar (Bar): The current price bar for the order's symbol and timeframe.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing a boolean indicating if the order was executed
            and the resulting Trade object if applicable.
        """
        execution_price = order.get_last_fill_price() or bar.close
        fill_size = order.get_remaining_size()

        # Check risk limits
        if not self.risk_manager.check_risk_limits(
            order,
            fill_size,
            self.account_manager.equity,
            self.position_manager.get_all_positions(),
        ):
            self._reject_order(order, "Risk limits breached")
            return False, None

        # Update position
        self.position_manager.update_position(order, execution_price, fill_size)

        # Update trade
        trade = self.trade_manager.manage_trade(order, execution_price, fill_size, bar)

        # Update account
        cost = execution_price * fill_size
        commission = cost * self.trade_manager.commission_rate
        self.account_manager.update_cash(
            -cost - commission, f"Order execution for {order.details.ticker}"
        )

        # Handle order groups
        self.order_manager.handle_order_update(order)

        logger_main.info(
            f"Executed order: {order.id}, Symbol: {order.details.ticker}, "
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

    def _update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Update all open trades based on current market data.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        self.trade_manager.update_open_trades(market_data)

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

        unrealized_pnl = self.position_manager.get_total_unrealized_pnl()
        self.account_manager.update_equity(unrealized_pnl)
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
        equity = self.account_manager.equity
        current_prices = {
            symbol: data[min(data.keys())].close for symbol, data in market_data.items()
        }

        asset_value = self.position_manager.get_long_position_value(current_prices)
        liabilities = self.position_manager.get_short_position_value(current_prices)
        open_pnl = self.position_manager.get_total_unrealized_pnl()
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

        self.metrics = pd.concat([self.metrics, new_metrics], ignore_index=True)

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

    def create_order(self, order_details: OrderDetails) -> Optional[Order]:
        """
        Create a new order and add it to the order manager.

        Args:
            order_details (OrderDetails): The details of the order to be created.

        Returns:
            Optional[Order]: The created Order object, or None if margin requirements are not met.
        """
        if self._check_margin_requirements(order_details):
            return self.order_manager.create_order(order_details)
        else:
            logger_main.warning(f"Insufficient margin to create order: {order_details}")
            return None

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
            return self.order_manager.create_oco_order(order_details)
        elif isinstance(order_details, OCAOrderDetails):
            return self.order_manager.create_oca_order(order_details)
        elif isinstance(order_details, BracketOrderDetails):
            return self.order_manager.create_bracket_order(order_details)
        else:
            raise ValueError(f"Invalid order details type: {type(order_details)}")

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
        order = Order(
            str(uuid.uuid4()), order_details
        )  # Create a temporary Order object
        account_value = self.account_manager.equity
        current_positions = self.position_manager.get_all_positions()

        if self.risk_manager.check_margin_requirements(
            order, account_value, current_positions
        ):
            logger_main.info(
                f"Margin and risk checks passed for order: {order_details}"
            )
            return True
        else:
            logger_main.warning(f"Order failed margin or risk checks: {order_details}")
            return False

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
            Order.Direction.SELL if current_position > 0 else Order.Direction.BUY
        )
        order_details = OrderDetails(
            ticker=symbol,
            direction=direction,
            size=abs(size),
            price=None,  # Market order
            exectype=Order.ExecType.MARKET,
            timestamp=self.engine._current_timestamp,
            timeframe=self.engine.default_timeframe,
            strategy_id="CLOSE_POSITION",
        )
        return self.create_order(order_details)

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

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state.
        """
        return {
            "cash": self.account_manager.cash,
            "equity": self.account_manager.equity,
            "margin_used": self.account_manager.margin_used,
            "buying_power": self.account_manager.get_buying_power(),
            "realized_pnl": self.account_manager.realized_pnl,
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "positions": self.position_manager.positions,
            "open_trades": self.trade_manager.get_open_trades(),
            "pending_orders": self.order_manager.get_pending_orders(),
        }

    def get_metrics(self) -> pd.DataFrame:
        """
        Get the portfolio metrics.

        Returns:
            pd.DataFrame: The portfolio metrics dataframe.
        """
        return self.metrics

    def close_all_positions(self, timestamp: datetime) -> None:
        """
        Close all open positions in the portfolio.

        Args:
            timestamp (datetime): The current timestamp for order creation and processing.
        """
        for symbol, position in self.position_manager.positions.items():
            if position != ExtendedDecimal("0"):
                close_order = self._create_market_order_to_close(symbol, abs(position))
                self._execute_order(close_order, self.engine.get_current_bar(symbol))

        logger_main.info("All positions have been closed.")

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics using the Reporter.

        Returns:
            Dict[str, float]: A dictionary of performance metrics.
        """
        if self.reporter:
            return self.reporter.calculate_performance_metrics()
        else:
            logger_main.warning(
                "No Reporter set. Unable to calculate performance metrics."
            )
            return {}
