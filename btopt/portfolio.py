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

    to specialized manager classes and maintaining overall portfolio state.
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
        self.account_manager = AccountManager(initial_capital, margin_ratio)
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(**risk_manager_config)

        self.reporter: Optional[Reporter] = None
        self._symbol_weights: Dict[str, ExtendedDecimal] = {}
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

        self._initialize_symbol_weights()

    def update(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the portfolio state based on current market data.

        This method coordinates updates across all manager components.

        Args:
            timestamp: The current timestamp.
            market_data: The current market data for all symbols and timeframes.
        """
        # Process orders
        filled_orders = self.order_manager.process_orders(timestamp, market_data)

        # Handle filled orders
        for order in filled_orders:
            self._handle_filled_order(order, market_data)

        # Update positions and account
        self._update_positions_and_account(market_data)

        # Update metrics
        self._update_metrics(timestamp)

        # Check for margin call
        if self.risk_manager.check_margin_call(
            self.account_manager.equity, self.account_manager.margin_used
        ):
            self._handle_margin_call(market_data)

    def _handle_filled_order(
        self, order: Order, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Handle a filled order by updating positions, trades, and account state.

        Args:
            order: The filled order.
            market_data: The current market data.
        """
        symbol = order.details.ticker
        if not order.details.timeframe:
            order.details.timeframe = self.engine.default_timeframe

        current_bar = market_data[symbol][order.details.timeframe]
        execution_price = order.get_last_fill_price() or current_bar.close

        # Update position
        self.position_manager.update_position(order, execution_price)

        # Update account
        cost = execution_price * order.get_filled_size()
        commission = cost * self.trade_manager.commission_rate
        self.account_manager.update_cash(
            -cost - commission, f"Order execution for {symbol}"
        )

        # Create or update trade
        if order.details.direction == Order.Direction.LONG:
            self.trade_manager.create_trade(order, execution_price, current_bar)
        else:
            self._close_trades(order, execution_price, current_bar)

    def _close_trades(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> None:
        """
        Close trades based on the given order.

        Args:
            order: The order that triggers trade closure.
            execution_price: The execution price for closing trades.
            bar: The current price bar.
        """
        remaining_size = order.get_filled_size()
        symbol = order.details.ticker
        for trade in self.trade_manager.get_trades_for_symbol(symbol):
            if remaining_size >= trade.current_size:
                self.trade_manager.close_trade(trade, order, execution_price, bar)
                remaining_size -= trade.current_size
            else:
                self.trade_manager.partial_close_trade(
                    trade, order, execution_price, bar, remaining_size
                )
                break

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

    def _update_metrics(self, timestamp: datetime) -> None:
        """
        Update portfolio metrics.

        Args:
            timestamp: The current timestamp.
        """
        equity = self.account_manager.equity
        asset_value = self.position_manager.get_long_position_value(
            self._get_current_prices()
        )
        liabilities = self.position_manager.get_short_position_value(
            self._get_current_prices()
        )

        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "cash": [self.account_manager.cash],
                "equity": [equity],
                "asset_value": [asset_value],
                "liabilities": [liabilities],
                "open_pnl": [self.position_manager.get_total_unrealized_pnl()],
                "closed_pnl": [self.position_manager.get_total_realized_pnl()],
                "portfolio_return": [self._calculate_return(equity)],
            }
        )

        if self.metrics.empty:
            self.metrics = new_row
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True).fillna(0)

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

    def _handle_margin_call(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Handle a margin call by closing positions as necessary.

        Args:
            market_data: The current market data for all symbols and timeframes.
        """
        current_prices = self._get_current_prices()
        positions_to_close = self.risk_manager.handle_margin_call(
            self.position_manager.get_all_positions(), current_prices
        )

        for order in positions_to_close:
            self._handle_filled_order(order, market_data)

    def _get_current_prices(self) -> Dict[str, ExtendedDecimal]:
        """
        Get the current prices for all symbols in the portfolio.

        Returns:
            A dictionary mapping symbols to their current prices.
        """
        return {
            symbol: self.engine.get_current_data(symbol).close
            for symbol in self._symbol_weights
        }

    def create_order(self, order_details: OrderDetails) -> Order:
        """
        Create a new order.

        Args:
            order_details: The details of the order to be created.

        Returns:
            The created Order object.
        """
        return self.order_manager.create_order(order_details)

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
    ) -> Tuple[Order, Order, Order, BracketGroup]:
        """
        Create a Bracket order.

        Args:
            bracket_details: A BracketOrderDetails object containing the details for entry, take profit, and stop loss orders.

        Returns:
            A tuple containing the entry order, take profit order, stop loss order, and the BracketGroup.
        """
        entry_order = self.create_order(bracket_details.entry_order)
        take_profit_order = self.create_order(bracket_details.take_profit_order)
        stop_loss_order = self.create_order(bracket_details.stop_loss_order)
        bracket_group = self.order_manager.create_bracket_group(
            entry_order, take_profit_order, stop_loss_order
        )
        return entry_order, take_profit_order, stop_loss_order, bracket_group

    def create_complex_order(
        self,
        order_details: Union[OCOOrderDetails, OCAOrderDetails, BracketOrderDetails],
    ) -> Union[
        Tuple[Order, Order, OCOGroup],
        Tuple[List[Order], OCAGroup],
        Tuple[Order, Order, Order, BracketGroup],
    ]:
        """
        Create a complex order based on the type of order details provided.

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
            raise ValueError(f"Invalid order details type: {type(order_details)}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            True if the order was successfully cancelled, False otherwise.
        """
        return self.order_manager.cancel_order(order_id)

    def close_all_positions(self, timestamp: datetime) -> None:
        """
        Close all open positions in the portfolio.

        This method delegates the task of closing all positions to the PositionManager,
        and then processes the resulting orders.

        Args:
            timestamp (datetime): The current timestamp for order creation and processing.
        """
        close_orders = self.position_manager.generate_close_all_orders(
            timestamp,
            self.engine.default_timeframe,
            self._get_current_prices(),
        )

        for order in close_orders:
            self._handle_filled_order(order, self._create_market_data_for_order(order))

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
            symbol: The symbol to check.

        Returns:
            The current position size (positive for long, negative for short).
        """
        position = self.position_manager.get_position(symbol)
        return position.quantity if position else ExtendedDecimal("0")

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

    def get_open_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        """
        Get all open trades, optionally filtered by strategy ID.

        Args:
            strategy_id: The ID of the strategy to filter trades for.

        Returns:
            A list of open trades.
        """
        return self.trade_manager.get_open_trades(strategy_id)

    def get_closed_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        """
        Get all closed trades, optionally filtered by strategy ID.

        Args:
            strategy_id: The ID of the strategy to filter trades for.

        Returns:
            A list of closed trades.
        """
        return self.trade_manager.get_closed_trades(strategy_id)

    def get_all_positions(self) -> Dict[str, ExtendedDecimal]:
        """
        Get all current positions.

        Returns:
            A dictionary mapping symbols to their current position sizes.
        """
        return self.position_manager.get_all_positions()

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            A dictionary containing the current portfolio state.
        """
        return {
            "cash": self.account_manager.cash,
            "equity": self.account_manager.equity,
            "long_position_value": self.position_manager.get_long_position_value(
                self._get_current_prices()
            ),
            "short_position_value": self.position_manager.get_short_position_value(
                self._get_current_prices()
            ),
            "open_trades": self.trade_manager.get_open_trades(),
            "closed_trades_count": len(self.trade_manager.get_closed_trades()),
            "pending_orders": self.order_manager.get_active_orders(),
            "realized_pnl": self.position_manager.get_total_realized_pnl(),
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "margin_used": self.account_manager.margin_used,
            "buying_power": self.account_manager.buying_power,
            "positions": self.position_manager.get_all_positions(),
        }

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

    def calculate_risk_amount(self, symbol: str) -> ExtendedDecimal:
        """
        Calculate the risk amount for a specific symbol.

        Args:
            symbol: The symbol to calculate the risk amount for.

        Returns:
            The calculated risk amount.
        """
        return self.risk_manager.calculate_position_size(
            self.account_manager.equity,
            self._get_current_prices()[symbol],
            self.risk_manager.max_risk_per_trade,
        )

    def _initialize_symbol_weights(self) -> None:
        symbols = self.engine._dataview.symbols
        weight = ExtendedDecimal("1") / ExtendedDecimal(str(len(symbols)))
        for symbol in symbols:
            self._symbol_weights[symbol] = weight

    def set_symbol_weight(self, symbol: str, weight: float) -> None:
        """
        Set the weight for a specific symbol in the portfolio.

        Args:
            symbol: The symbol to set the weight for.
            weight: The new weight for the symbol (between 0 and 1).

        Raises:
            ValueError: If the weight is not between 0 and 1 or if the symbol is not in the portfolio.
        """
        if symbol not in self._symbol_weights:
            raise ValueError(f"Symbol {symbol} not found in portfolio")
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")

        self._symbol_weights[symbol] = ExtendedDecimal(str(weight))
        self._normalize_weights()

    def get_symbol_weight(self, symbol: str) -> ExtendedDecimal:
        """
        Get the weight of a specific symbol in the portfolio.

        Args:
            symbol: The symbol to get the weight for.

        Returns:
            The weight of the symbol.

        Raises:
            KeyError: If the symbol is not found in the portfolio.
        """
        if symbol not in self._symbol_weights:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
        return self._symbol_weights[symbol]

    def set_all_symbol_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for all symbols in the portfolio.

        Args:
            weights: A dictionary mapping symbols to their weights.

        Raises:
            ValueError: If any weight is not between 0 and 1 or if any symbol is not in the portfolio.
        """
        for symbol, weight in weights.items():
            if symbol not in self._symbol_weights:
                raise ValueError(f"Symbol {symbol} not found in portfolio")
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for {symbol} must be between 0 and 1")
            self._symbol_weights[symbol] = ExtendedDecimal(str(weight))

        self._normalize_weights()

    def get_all_symbol_weights(self) -> Dict[str, ExtendedDecimal]:
        """
        Get the weights of all symbols in the portfolio.

        Returns:
            A dictionary mapping symbols to their weights.
        """
        return self._symbol_weights.copy()

    def _normalize_weights(self) -> None:
        """
        Normalize the symbol weights to ensure they sum to 1 (100%).
        """
        total_weight = sum(self._symbol_weights.values())
        if total_weight > 0:
            for symbol in self._symbol_weights:
                self._symbol_weights[symbol] /= total_weight

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

    def handle_order_group_events(self) -> None:
        """
        Handle events from order groups (e.g., when an OCO order is filled).
        """
        self.order_manager.handle_order_group_events()

    def update_risk_parameters(self, **kwargs: Any) -> None:
        """
        Update risk management parameters.

        Args:
            **kwargs: Risk management parameters to update.
        """
        self.risk_manager.update_parameters(**kwargs)

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
                logger_main.error(
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
            timestamp=datetime.now(),
            timeframe=self.engine.default_timeframe,
            strategy_id="MARGIN_CALL",
        )
        return self.create_order(order_details)

    def _notify_margin_call_status(self) -> None:
        """Notify about the margin call status. This could be extended to send emails, alerts, etc."""
        # Implementation depends on your notification system
        pass

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
