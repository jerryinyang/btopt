from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..btopt.data.bar import Bar
from ..btopt.data.timeframe import Timeframe
from ..btopt.order import Order, OrderDetails
from ..btopt.trade import Trade
from ..btopt.types import EngineType
from ..btopt.util.ext_decimal import ExtendedDecimal
from ..btopt.util.log_config import logger_main


class Portfolio:
    def __init__(
        self,
        engine: EngineType,
        initial_capital: ExtendedDecimal = ExtendedDecimal("100000"),
        commission_rate: ExtendedDecimal = ExtendedDecimal("0.001"),
        slippage: ExtendedDecimal = ExtendedDecimal("0"),
        pyramiding: int = 1,
        margin_ratio: ExtendedDecimal = ExtendedDecimal("0.5"),
        margin_call_threshold: ExtendedDecimal = ExtendedDecimal("0.3"),
        risk_percentage: ExtendedDecimal = ExtendedDecimal("0.02"),
    ):
        """
        Initialize the Portfolio with given parameters and engine.

        Args:
            engine (EngineType): The Engine instance managing the trading system.
            initial_capital (ExtendedDecimal): The starting capital of the portfolio.
            commission_rate (ExtendedDecimal): The commission rate for trades.
            slippage (ExtendedDecimal): The slippage rate for trades.
            pyramiding (int): The maximum number of allowed positions per symbol.
            margin_ratio (ExtendedDecimal): The required margin ratio for trades.
            margin_call_threshold (ExtendedDecimal): The threshold for triggering a margin call.
            risk_percentage (ExtendedDecimal): The default risk percentage for the portfolio.
        """
        self.engine = engine
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.pyramiding = pyramiding
        self.margin_ratio = margin_ratio
        self.margin_call_threshold = margin_call_threshold
        self.risk_percentage = risk_percentage

        self.long_position_value = ExtendedDecimal("0")
        self.short_position_value = ExtendedDecimal("0")
        self.buying_power = initial_capital

        self.positions: Dict[str, ExtendedDecimal] = {}
        self.avg_entry_prices: Dict[str, ExtendedDecimal] = {}
        self.open_trades: Dict[str, List[Trade]] = {}
        self.closed_trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.trade_count = 0
        self.margin_used = ExtendedDecimal("0")
        self.limit_exit_orders: List[Order] = []
        self.transaction_log: List[Dict[str, Any]] = []

        self.current_market_data: Dict[str, Dict[Timeframe, Bar]] = {}

        self.updated_orders: List[Order] = []
        self.updated_trades: List[Trade] = []

        # Initialize symbol weights
        self._symbol_weights: Dict[str, ExtendedDecimal] = {}
        self._initialize_symbol_weights()

        # Initialize the metrics DataFrame with new columns
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

        self.realized_pnl_since_last_update = ExtendedDecimal("0")

    # region Initialization and Configuration

    def _initialize_symbol_weights(self) -> None:
        """
        Initialize the symbol weights based on the symbols provided by the engine.

        This method sets equal weights for all symbols in the engine's DataView.
        """
        symbols = self.engine._dataview.symbols
        weight = ExtendedDecimal("1") / ExtendedDecimal(str(len(symbols)))
        for symbol in symbols:
            self._symbol_weights[symbol] = weight

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the portfolio.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self.initial_capital = ExtendedDecimal(
            str(config.get("initial_capital", self.initial_capital))
        )
        self.commission_rate = ExtendedDecimal(
            str(config.get("commission_rate", self.commission_rate))
        )
        self.slippage = ExtendedDecimal(str(config.get("slippage", self.slippage)))
        self.pyramiding = config.get("pyramiding", self.pyramiding)
        self.margin_ratio = ExtendedDecimal(
            str(config.get("margin_ratio", self.margin_ratio))
        )
        self.margin_call_threshold = ExtendedDecimal(
            str(config.get("margin_call_threshold", self.margin_call_threshold))
        )
        self.risk_percentage = ExtendedDecimal(
            str(config.get("risk_percentage", self.risk_percentage))
        )

    def reset(self) -> None:
        """
        Reset the portfolio to its initial state.
        """
        self.__init__(
            self.initial_capital,
            self.commission_rate,
            self.slippage,
            self.pyramiding,
            self.margin_ratio,
            self.margin_call_threshold,
            self.engine,
        )
        logger_main.info("Portfolio reset to initial state.")

    # endregion

    # region Portfolio Update and Market Data Processing

    def update(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update the portfolio state based on the current market data.

        This method processes pending orders, updates open trades, and refreshes portfolio metrics.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.

        Side effects:
            - Updates self.current_market_data
            - Processes pending orders
            - Updates open trades
            - Updates portfolio metrics
            - May trigger margin calls
        """
        self.current_market_data = market_data

        self._process_pending_orders(timestamp, market_data)
        self._update_open_trades(market_data)
        self._update_metrics(timestamp)
        self._update_margin_and_buying_power()

        if self._check_margin_call():
            self._handle_margin_call()

    def _update_metrics(self, timestamp: datetime) -> None:
        """
        Update the portfolio metrics based on the current state.

        This method calculates and stores essential metrics including cash, equity,
        asset value, liabilities, open/closed PnL, and portfolio returns.

        Args:
            timestamp (datetime): The current timestamp for the metrics update.

        Side effects:
            - Updates self.metrics DataFrame
        """
        realized_pnl, unrealized_pnl = self._calculate_pnl()

        # Add the accumulated realized_pnl to the total realized_pnl
        realized_pnl += self.realized_pnl_since_last_update

        equity = self.calculate_equity()

        if not self.metrics.empty:
            previous_equity = self.metrics.iloc[-1]["equity"]
            # Adjust the equity change by the realized_pnl since last update
            equity_change = (
                equity - previous_equity + self.realized_pnl_since_last_update
            )
            portfolio_return = equity_change / previous_equity
        else:
            portfolio_return = ExtendedDecimal("0")

        new_row = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "cash": [self.cash],
                "equity": [equity],
                "asset_value": [self.long_position_value],
                "liabilities": [self.short_position_value],
                "realized_pnl": [realized_pnl],
                "unrealized_pnl": [unrealized_pnl],
                "portfolio_return": [portfolio_return],
            }
        )

        # Reset the accumulated realized_pnl
        self.realized_pnl_since_last_update = ExtendedDecimal("0")

        # Avoid FutureWarning
        if self.metrics.empty:
            self.metrics = new_row
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)

    def _get_current_price(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current price for a symbol.

        Args:
            symbol (str): The symbol to get the price for.

        Returns:
            ExtendedDecimal: The current price of the symbol.
        """
        if self.engine and hasattr(self.engine, "_current_market_data"):
            market_data = self.engine._current_market_data[symbol]
            current_data = market_data[min(market_data.keys())]
            if current_data is not None:
                return ExtendedDecimal(str(current_data["close"]))

        # If we can't get the current price, use the last known price
        logger_main.log_and_raise(
            ValueError(f"Unable to get current price for {symbol} {self.engine}")
        )

    def _update_cash(self, amount: ExtendedDecimal, reason: str) -> None:
        """
        Update the cash balance of the portfolio.

        Args:
            amount (ExtendedDecimal): The amount to add (positive) or subtract (negative) from the cash balance.
            reason (str): The reason for the cash update (e.g., "Trade execution", "Commission", "Dividend").

        Side effects:
            - Updates self.cash
            - Logs the transaction
        """
        self.cash += amount
        self._log_transaction("Cash", amount, reason)

    def _update_margin_and_buying_power(self) -> None:
        """
        Update the margin used and buying power based on current positions and equity.

        This method recalculates the margin used based on all open positions and
        updates the buying power accordingly.

        Side effects:
            - Updates self.margin_used
            - Updates self.buying_power
        """
        self.margin_used = ExtendedDecimal("0")
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            position_value = abs(position) * current_price
            if position > ExtendedDecimal("0"):
                self.margin_used += position_value * self.margin_ratio
            else:  # SHORT
                self.margin_used += (
                    position_value * self.margin_ratio * ExtendedDecimal("2")
                )

        equity = self.calculate_equity()
        self.buying_power = (equity - self.margin_used) / self.margin_ratio

    # endregion

    # region Risk Amount Management
    def calculate_risk_amount(self, percentage: float = 1.0) -> ExtendedDecimal:
        """
        Calculate the total risk amount based on available equity and margin requirements.

        This method computes the total amount that can be risked in trading, considering
        the current equity and margin requirements. It allows for specifying a percentage
        of this total to be used.

        Args:
            percentage (float): The percentage of the total available equity to use.
                                Defaults to 1.0 (100%).

        Returns:
            ExtendedDecimal: The calculated risk amount.

        Raises:
            ValueError: If the percentage is not between 0 and 1.
        """
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")

        total_equity = self.calculate_equity()
        available_margin = self.get_available_margin()

        # Use the lesser of total equity and available margin to be conservative
        base_risk_amount = min(total_equity, available_margin)

        return base_risk_amount * ExtendedDecimal(str(percentage))

    def set_symbol_weight(self, symbol: str, weight: float) -> None:
        """
        Set the weight for a specific symbol in the portfolio.

        This method updates the weight of a given symbol and recalculates all weights
        to ensure they sum to 1 (100%).

        Args:
            symbol (str): The symbol to set the weight for.
            weight (float): The new weight for the symbol (between 0 and 1).

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
            symbol (str): The symbol to get the weight for.

        Returns:
            ExtendedDecimal: The weight of the symbol.

        Raises:
            KeyError: If the symbol is not found in the portfolio.
        """
        if symbol not in self._symbol_weights:
            raise KeyError(f"Symbol {symbol} not found in portfolio")
        return self._symbol_weights[symbol]

    def set_all_symbol_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for all symbols in the portfolio.

        This method updates the weights for all provided symbols and recalculates
        the weights to ensure they sum to 1 (100%).

        Args:
            weights (Dict[str, float]): A dictionary mapping symbols to their weights.

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
            Dict[str, ExtendedDecimal]: A dictionary mapping symbols to their weights.
        """
        return self._symbol_weights.copy()

    def get_risk_amount_for_symbol(self, symbol: str) -> ExtendedDecimal:
        """
        Calculate the risk amount for a specific symbol based on its weight.

        This method computes the portion of the total risk amount allocated to a
        specific symbol, based on the symbol's weight in the portfolio.

        Args:
            symbol (str): The symbol to calculate the risk amount for.

        Returns:
            ExtendedDecimal: The calculated risk amount for the symbol.

        Raises:
            KeyError: If the symbol is not found in the portfolio.
        """
        if symbol not in self._symbol_weights:
            raise KeyError(f"Symbol {symbol} not found in portfolio")

        total_risk_amount = self.calculate_risk_amount()
        symbol_weight = self._symbol_weights[symbol]

        return total_risk_amount * symbol_weight

    def _normalize_weights(self) -> None:
        """
        Normalize the symbol weights to ensure they sum to 1 (100%).

        This private method is called after weight updates to maintain the
        integrity of the weight distribution across all symbols.
        """
        total_weight = sum(self._symbol_weights.values())
        if total_weight == 0:
            return  # Avoid division by zero

        for symbol in self._symbol_weights:
            self._symbol_weights[symbol] /= total_weight

    # endregion

    # region Order and Trade Management

    def create_order(
        self,
        symbol: str,
        direction: Order.Direction,
        size: float,
        order_type: Order.ExecType,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Order:
        """
        Create and add a new order to pending orders.

        Args:
            symbol (str): The symbol for the order.
            direction (Order.Direction): The direction of the order (LONG or SHORT).
            size (float): The size of the order.
            order_type (Order.ExecType): The type of the order (e.g., MARKET, LIMIT).
            price (Optional[float]): The price for limit orders.
            **kwargs: Additional order parameters.

        Returns:
            Order: The created order object.
        """
        order_details = OrderDetails(
            ticker=symbol,
            direction=direction,
            size=ExtendedDecimal(str(size)),
            price=ExtendedDecimal(str(price)) if price is not None else None,
            exectype=order_type,
            timestamp=datetime.now(),
            slippage=self.slippage,
            **kwargs,
        )
        order = Order(order_id=self._generate_order_id(), details=order_details)
        self.add_pending_order(order)
        return order

    def execute_order(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Tuple[bool, Optional[Trade]]:
        """
        Execute an order and update the portfolio accordingly.

        This method handles the execution of an order, updating cash, creating or updating trades,
        and managing position changes.

        Args:
            order (Order): The order to execute.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Tuple[bool, Optional[Trade]]: A tuple containing:
                - bool: True if the order was executed successfully, False otherwise.
                - Optional[Trade]: The resulting Trade object if applicable, None otherwise.

        Side effects:
            - Calls self._update_cash
            - Calls self._manage_trade
            - Calls self._update_margin_and_buying_power
            - May update self.updated_orders
        """
        symbol = order.details.ticker
        size = order.details.size
        direction = order.details.direction

        cost = execution_price * size
        commission = cost * self.commission_rate

        # Check margin requirements
        if not self._check_margin_requirements(order, cost):
            logger_main.warning(f"Insufficient margin to execute order: {order}")
            return False, None

        # Update cash
        cash_change = (
            -cost - commission
            if direction == Order.Direction.LONG
            else cost - commission
        )
        self._update_cash(cash_change, f"Order execution for {symbol}")

        # Manage trade
        trade = self._manage_trade(order, execution_price, bar)

        # Update margin and buying power
        self._update_margin_and_buying_power()

        # Update order status
        order.status = Order.Status.FILLED
        self.updated_orders.append(order)

        logger_main.info(f"Executed order: {order}, resulting trade: {trade}")
        return True, trade

    def add_pending_order(self, order: Order) -> None:
        """
        Add an order to the appropriate list of pending orders.

        Args:
            order (Order): The order to add to pending orders.
        """
        if (
            order.family_role == Order.FamilyRole.CHILD_EXIT
            and order.details.exectype
            in [Order.ExecType.EXIT_LIMIT, Order.ExecType.EXIT_STOP]
        ):
            self.limit_exit_orders.append(order)
        else:
            self.pending_orders.append(order)
        logger_main.info(f"Added pending order: {order}")

    def cancel_order(self, order: Order) -> bool:
        """
        Cancel an order and update its status.

        Args:
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        elif order in self.limit_exit_orders:
            self.limit_exit_orders.remove(order)
        else:
            logger_main.warning(
                f"Failed to cancel order (not found in pending orders): {order}",
            )
            return False

        order.status = Order.Status.CANCELED
        self.updated_orders.append(order)
        logger_main.info(f"Cancelled order: {order}")
        return True

    def modify_order(self, order_id: int, new_details: Dict[str, Any]) -> bool:
        """
        Modify an existing pending order.

        Args:
            order_id (int): The ID of the order to modify.
            new_details (Dict[str, Any]): A dictionary containing the new details for the order.

        Returns:
            bool: True if the order was successfully modified, False otherwise.
        """
        for order in self.pending_orders:
            if order.id == order_id:
                for key, value in new_details.items():
                    if hasattr(order.details, key):
                        setattr(order.details, key, value)
                self.updated_orders.append(order)
                logger_main.info(f"Modified order: {order}")
                return True
        logger_main.info(f"Order with ID {order_id} not found in pending orders.")
        return False

    def close_positions(self, strategy_id: str, symbol: Optional[str] = None) -> bool:
        """
        Close positions for a specific strategy and/or symbol.

        Args:
            strategy_id (str): The ID of the strategy.
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.

        Returns:
            bool: True if any positions were closed, False otherwise.
        """
        closed_any = False
        trades_to_close = self._get_open_trades(strategy_id, symbol)

        current_price = self.current_market_data[symbol][
            min(self.engine._dataview.data[symbol].keys())
        ]
        for trade in trades_to_close:
            self.close_trade(trade, current_price)
            closed_any = True

        if closed_any:
            logger_main.info(
                f"Closed positions for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
            )
        else:
            logger_main.info(
                f"No positions to close for strategy {strategy_id}"
                + (f" and symbol {symbol}" if symbol else ""),
            )

        return closed_any

    def close_all_positions(self, current_timestamp: datetime) -> None:
        """
        Close all open positions in the portfolio.

        Args:
            current_timestamp (datetime): The current timestamp to use for closing trades.
        """
        symbols_to_close = list(self.open_trades.keys())

        for symbol in symbols_to_close:
            trades_to_close = self.open_trades[symbol][:]  # Create a copy of the list
            for trade in trades_to_close:
                current_price = self._get_current_price(symbol)
                self.close_trade(trade, current_price)

        logger_main.info("Closed all open positions.")

    def close_trade(
        self,
        trade: Trade,
        current_price: ExtendedDecimal,
        order: Optional[Order] = None,
    ) -> None:
        """
        Close a specific trade and update the portfolio accordingly.

        Args:
            trade (Trade): The trade to close.
            current_price (ExtendedDecimal): The current market price to close the trade at.
            order (Optional[Order]): The order that triggered the trade closure, if any.

        Side effects:
            - Calls self._update_cash
            - Calls self._update_positions
            - Moves trade from self.open_trades to self.closed_trades
            - Calls self._update_margin_and_buying_power
            - Updates self.updated_trades
        """
        symbol = trade.ticker
        close_size = trade.current_size
        close_value = close_size * current_price
        commission = close_value * self.commission_rate

        # Update cash
        cash_change = (
            close_value - commission
            if trade.direction == Order.Direction.LONG
            else -close_value - commission
        )
        self._update_cash(cash_change, f"Trade closure for {symbol}")

        # Update positions
        position_change = (
            -close_size if trade.direction == Order.Direction.LONG else close_size
        )
        self._update_positions(symbol, position_change, current_price)

        # Close the trade
        trade.close(
            order or self._generate_close_order(trade, current_price),
            current_price,
            self._create_dummy_bar(current_price, trade),
        )
        # Calculate and accumulate realized PnL
        self.realized_pnl_since_last_update += trade.metrics.pnl()
        logger_main.warning(f"REALIZED PROFIT: {self.realized_pnl_since_last_update}")

        # Move trade from open to closed
        self.open_trades[symbol].remove(trade)
        if not self.open_trades[symbol]:
            del self.open_trades[symbol]
        self.closed_trades.append(trade)

        # Update margin and buying power
        self._update_margin_and_buying_power()

        # Add to updated trades
        self._add_to_updated_trades(trade)

        logger_main.info(f"Closed trade: {trade}")

    def _process_pending_orders(
        self, timestamp: datetime, market_data: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Process all pending orders based on current market data.

        Args:
            timestamp (datetime): The current timestamp.
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data.

        Side effects:
            - May execute orders
            - May cancel expired orders
            - Updates self.pending_orders and self.limit_exit_orders
        """
        orders_to_process = self.pending_orders[:] + self.limit_exit_orders[:]

        for order in orders_to_process:
            symbol = order.details.ticker
            timeframe = order.details.timeframe or min(market_data[symbol].keys())

            try:
                current_bar = market_data[symbol][timeframe]
            except KeyError:
                logger_main.warning(
                    f"No market data for {symbol} at timeframe {timeframe}. Skipping order."
                )
                continue

            is_filled, fill_price = order.is_filled(current_bar)
            if is_filled:
                executed, _ = self.execute_order(order, fill_price, current_bar)
                if executed:
                    self._remove_executed_order(order)
            elif order.is_expired(timestamp):
                self._cancel_expired_order(order)

    def _manage_trade(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Optional[Trade]:
        """
        Manage trade creation, updates, closures, and reversals.

        This method handles all aspects of trade management, including creating new trades,
        updating existing trades, closing trades, and handling trade reversals.

        Args:
            order (Order): The order being executed.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Optional[Trade]: The newly created or updated trade, if applicable.

        Side effects:
            - May create new trades in self.open_trades
            - May update existing trades in self.open_trades
            - May close trades and move them to self.closed_trades
            - Updates self.trade_count
            - Calls self._update_positions
            - Adds affected trades to self.updated_trades
        """
        symbol = order.details.ticker
        size = order.get_filled_size()
        direction = order.details.direction

        existing_position = self.positions.get(symbol, ExtendedDecimal("0"))
        is_reversal = (
            existing_position > ExtendedDecimal("0")
            and direction == Order.Direction.SHORT
        ) or (
            existing_position < ExtendedDecimal("0")
            and direction == Order.Direction.LONG
        )

        if is_reversal:
            # Close existing position
            self._close_position(symbol, existing_position, execution_price, bar, order)
            remaining_size = size - abs(existing_position)
            if remaining_size > ExtendedDecimal("0"):
                # Open new position in opposite direction
                return self._create_new_trade(
                    order, execution_price, bar, remaining_size
                )
        else:
            return self._create_new_trade(order, execution_price, bar, size)

        return None

    def _calculate_pnl(self) -> Tuple[ExtendedDecimal, ExtendedDecimal]:
        """
        Calculate the realized and unrealized Profit and Loss (PnL) for the portfolio.

        Returns:
            Tuple[ExtendedDecimal, ExtendedDecimal]: A tuple containing:
                - ExtendedDecimal: The realized PnL
                - ExtendedDecimal: The unrealized PnL
        """
        realized_pnl = sum(trade.metrics.pnl for trade in self.closed_trades)
        unrealized_pnl = ExtendedDecimal("0")

        for trades in self.open_trades.values():
            for trade in trades:
                unrealized_pnl += trade.metrics.unrealized_pnl

        return realized_pnl, unrealized_pnl

    def _log_transaction(
        self, transaction_type: str, amount: ExtendedDecimal, details: str
    ) -> None:
        """
        Log a transaction to the transaction log.

        Args:
            transaction_type (str): The type of transaction (e.g., "Cash", "Position", "Trade").
            amount (ExtendedDecimal): The amount involved in the transaction.
            details (str): Additional details about the transaction.

        Side effects:
            - Appends a new entry to self.transaction_log
        """
        self.transaction_log.append(
            {
                "timestamp": datetime.now(),
                "type": transaction_type,
                "amount": amount,
                "details": details,
            }
        )

    def _create_new_trade(
        self,
        order: Order,
        execution_price: ExtendedDecimal,
        bar: Bar,
        size: ExtendedDecimal,
    ) -> Trade:
        """
        Create a new trade based on the executed order.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            bar (Bar): The current price bar.
            size (ExtendedDecimal): The size of the new trade.

        Returns:
            Trade: The newly created trade.

        Side effects:
            - Creates a new Trade object and adds it to self.open_trades
            - Increments self.trade_count
            - Calls self._update_positions
            - Adds the new trade to self.updated_trades
        """
        symbol = order.details.ticker
        self.trade_count += 1
        new_trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
        )
        new_trade.initial_size = size
        new_trade.current_size = size

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []
        self.open_trades[symbol].append(new_trade)

        self._update_positions(
            symbol,
            size if order.details.direction == Order.Direction.LONG else -size,
            execution_price,
        )
        self._add_to_updated_trades(new_trade)

        return new_trade

    def _close_position(
        self,
        symbol: str,
        position_size: ExtendedDecimal,
        execution_price: ExtendedDecimal,
        bar: Bar,
        order: Order,
    ) -> None:
        """
        Close an existing position for a given symbol.

        Args:
            symbol (str): The symbol of the position to close.
            position_size (ExtendedDecimal): The size of the position to close.
            execution_price (ExtendedDecimal): The price at which to close the position.
            bar (Bar): The current price bar.
            order (Order): The order that triggered the position closure.

        Side effects:
            - Updates or closes trades in self.open_trades
            - May move trades to self.closed_trades
            - Calls self._update_positions
            - Updates self.updated_trades
        """
        remaining_size = abs(position_size)
        for trade in self.open_trades.get(symbol, [])[
            :
        ]:  # Create a copy of the list to iterate
            if remaining_size <= ExtendedDecimal("0"):
                break

            if remaining_size >= trade.current_size:
                # Fully close this trade
                self.close_trade(trade, execution_price, order)
                remaining_size -= trade.current_size
            else:
                # Partially close this trade
                partial_close_size = remaining_size
                trade.partial_close(partial_close_size, execution_price, bar)
                self._add_to_updated_trades(trade)
                remaining_size = ExtendedDecimal("0")

        # Update positions
        self._update_positions(symbol, -position_size, execution_price)

    def _get_open_trades(
        self, strategy_id: str, symbol: Optional[str] = None
    ) -> List[Trade]:
        """
        Get open trades for a specific strategy and/or symbol.

        Args:
            strategy_id (str): The ID of the strategy.
            symbol (Optional[str]): The symbol to filter trades for. If None, return all open trades.

        Returns:
            List[Trade]: A list of open trades matching the criteria.
        """
        trades = []
        for sym, trade_list in self.open_trades.items():
            if symbol is None or sym == symbol:
                trades.extend([t for t in trade_list if t.strategy_id == strategy_id])
        return trades

    def _generate_close_order(
        self, trade: Trade, current_price: ExtendedDecimal
    ) -> Order:
        """
        Generate a closing order for a trade.

        Args:
            trade (Trade): The trade to close.
            current_price (ExtendedDecimal): The current market price for closing the trade.

        Returns:
            Order: A new Order object representing the closing order for the trade.
        """
        close_direction = (
            Order.Direction.SHORT
            if trade.direction == Order.Direction.LONG
            else Order.Direction.LONG
        )

        order_details = OrderDetails(
            ticker=trade.ticker,
            direction=close_direction,
            size=trade.current_size,
            price=current_price,
            exectype=Order.ExecType.MARKET,
            timestamp=datetime.now(),
            timeframe=trade.entry_bar.timeframe,
            strategy_id=trade.strategy_id,
        )

        return Order(order_id=self._generate_order_id(), details=order_details)

    def _remove_executed_order(self, order: Order) -> None:
        """Remove an executed order from the appropriate list."""
        if order in self.pending_orders:
            self.pending_orders.remove(order)
        elif order in self.limit_exit_orders:
            self.limit_exit_orders.remove(order)

    def _cancel_expired_order(self, order: Order) -> None:
        """Cancel an expired order and update its status."""
        self._remove_executed_order(order)
        order.status = Order.Status.CANCELED
        self.updated_orders.append(order)
        logger_main.info(
            f"Order {order.id} for {order.details.ticker} has expired and been canceled.",
        )

    def _update_positions(
        self, symbol: str, quantity: ExtendedDecimal, price: ExtendedDecimal
    ) -> None:
        """
        Update the positions for a given symbol.

        Args:
            symbol (str): The symbol of the asset.
            quantity (ExtendedDecimal): The quantity to add (positive) or subtract (negative) from the position.
            price (ExtendedDecimal): The price at which the position is being updated.

        Side effects:
            - Updates self.positions
            - Updates self.avg_entry_prices
            - Updates self.long_position_value and self.short_position_value
            - Logs the transaction
        """
        current_position = self.positions.get(symbol, ExtendedDecimal("0"))
        new_position = current_position + quantity

        if current_position == ExtendedDecimal("0"):
            self.avg_entry_prices[symbol] = price
        else:
            current_value = current_position * self.avg_entry_prices[symbol]
            new_value = abs(quantity) * price
            if new_position != ExtendedDecimal("0"):
                self.avg_entry_prices[symbol] = (current_value + new_value) / abs(
                    new_position
                )

        self.positions[symbol] = new_position

        if new_position == ExtendedDecimal("0"):
            del self.positions[symbol]
            del self.avg_entry_prices[symbol]

        # Update long_position_value and short_position_value
        if new_position > ExtendedDecimal("0"):
            self.long_position_value = new_position * price
            self.short_position_value = ExtendedDecimal("0")
        elif new_position < ExtendedDecimal("0"):
            self.short_position_value = abs(new_position) * price
            self.long_position_value = ExtendedDecimal("0")
        else:
            self.long_position_value = ExtendedDecimal("0")
            self.short_position_value = ExtendedDecimal("0")

        self._log_transaction("Position", quantity, f"Update for {symbol} at {price}")

    def _update_open_trades(self, market_data: Dict[str, Dict[Timeframe, Bar]]) -> None:
        """
        Update all open trades based on current market data.

        Args:
            market_data (Dict[str, Dict[Timeframe, Bar]]): The current market data for all symbols and timeframes.

        Side effects:
            - Updates trade metrics for all open trades
            - May add trades to self.updated_trades if their state changes
        """
        for symbol, trades in self.open_trades.items():
            timeframe = min(market_data[symbol].keys())
            current_bar = market_data[symbol][timeframe]
            for trade in trades:
                pre_update_state = trade.to_dict()
                trade.update(current_bar)
                if trade.to_dict() != pre_update_state:
                    self._add_to_updated_trades(trade)

    def _create_or_update_trade(
        self,
        order: Order,
        bar: Bar,
        size: Optional[ExtendedDecimal] = None,
    ) -> Trade:
        """
        Create a new trade based on the executed order.

        Args:
            order (Order): The executed order.
            execution_price (ExtendedDecimal): The price at which the order was executed.
            bar (Bar): The current price bar.
            size (Optional[ExtendedDecimal]): The size of the trade, if different from the order size.

        Returns:
            Trade: The newly created trade.
        """
        symbol = order.details.ticker
        trade_size = size or order.get_filled_size()

        if symbol not in self.open_trades:
            self.open_trades[symbol] = []

        self.trade_count += 1
        new_trade = Trade(
            trade_id=self.trade_count,
            entry_order=order,
            entry_bar=bar,
            commission_rate=self.commission_rate,
        )
        # Set the initial size of the trade
        new_trade.initial_size = trade_size
        new_trade.current_size = trade_size

        # Add new trade to
        self.open_trades[symbol].append(new_trade)

        logger_main.info(f"Created new trade: {new_trade}")
        return new_trade

    def _close_or_reduce_trade(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Tuple[List[Trade], ExtendedDecimal]:
        """
        Close or reduce existing trades based on the incoming order.

        This method processes the closure or reduction of trades for a given symbol.
        It handles both full and partial closures of trades, updating the portfolio
        state accordingly.

        Args:
            order (Order): The order associated with this closure or reduction.
            execution_price (ExtendedDecimal): The price at which to close or reduce the trades.
            bar (Bar): The current price bar.

        Returns:
            Tuple[List[Trade], ExtendedDecimal]: A tuple containing:
                - List[Trade]: A list of affected trades (closed or reduced).
                - ExtendedDecimal: The remaining size after closing or reducing trades.

        Side Effects:
            - Updates self.open_trades
            - Updates self.closed_trades
            - Calls self.close_trade for fully closed trades
            - Updates trade sizes for partially closed trades
            - Adds affected trades to self.updated_trades
        """
        symbol = order.details.ticker
        remaining_size = order.get_filled_size()

        for trade in self.open_trades.get(symbol, [])[:]:
            trade_size = trade.current_size
            # Create a copy of the list to iterate
            if remaining_size <= ExtendedDecimal("0"):
                break

            if remaining_size >= trade.current_size:
                # Fully close this trade
                self.close_trade(trade, execution_price, order)
                remaining_size -= trade_size

            else:
                # Partially close this trade
                partial_close_size = remaining_size

                # Create a dummy trade with the partially filled size
                closed_trade = Trade(
                    trade_id=self.trade_count + 1,
                    entry_order=trade.entry_order,
                    entry_bar=trade.entry_bar,
                    commission_rate=self.commission_rate,
                )
                closed_trade.initial_size = partial_close_size
                closed_trade.current_size = partial_close_size
                self.close_trade(closed_trade, execution_price, order)
                self.closed_trades.append(closed_trade)

                # Update the original trade
                trade.current_size -= partial_close_size
                self._add_to_updated_trades(trade)

                remaining_size = ExtendedDecimal("0")

        return remaining_size

    def _reverse_trade(
        self, order: Order, execution_price: ExtendedDecimal, bar: Bar
    ) -> Optional[Trade]:
        """
        Handle the creation of a new trade in the opposite direction after closing existing trades.

        This method processes trade reversal by closing existing trades in the opposite direction
        and creating a new trade if there's remaining size. It handles the complete lifecycle
        of a trade reversal, including position updates and trade notifications.

        Args:
            order (Order): The order that triggered the trade reversal.
            execution_price (ExtendedDecimal): The price at which the order is executed.
            bar (Bar): The current price bar.

        Returns:
            Optional[Trade]: The newly created trade in the opposite direction, or None if no new trade was created.

        Note:
            This method relies on `_close_or_reduce_trade` and `_create_or_update_trade` for trade management.
            It does not directly modify `self.positions` or update buying power, as these operations
            are handled by the called methods or higher-level methods.
        """
        # symbol = order.details.ticker
        # size = order.get_filled_size()
        # direction = order.details.direction

        # Close existing trades in the opposite direction
        remaining_size = self._close_or_reduce_trade(order, execution_price, bar)

        # If there's remaining size, create a new trade in the opposite direction
        new_trade = None
        if remaining_size > ExtendedDecimal("0"):
            new_trade = self._create_or_update_trade(order, bar, size=remaining_size)
            self._add_to_updated_trades(new_trade)

        return new_trade

    def _add_to_updated_trades(self, trade: Trade) -> None:
        """
        Add a trade to the updated_trades list if it's not already present.

        This method ensures that each trade is only added once to the updated_trades
        list during a single update cycle.

        Args:
            trade (Trade): The trade to be added to the updated_trades list.
        """
        if trade not in self.updated_trades:
            self.updated_trades.append(trade)
            logger_main.debug(f"Added trade {trade.id} to updated_trades list")

    def clear_updated_orders_and_trades(self) -> None:
        """
        Clear the lists of updated orders and trades.

        This method should be called by the Engine after notifying strategies
        of order and trade updates. It resets the updated_orders and updated_trades
        lists, preparing them for the next timestamp cycle.
        """
        self.updated_orders.clear()
        self.updated_trades.clear()
        logger_main.debug("Cleared updated orders and trades lists")

    # endregions

    # region Margin Related Methods

    def _check_margin_requirements(self, order: Order, cost: ExtendedDecimal) -> bool:
        """
        Check if there's sufficient margin to execute the order.

        Args:
            order (Order): The order to check.
            cost (ExtendedDecimal): The cost of the order.

        Returns:
            bool: True if there's sufficient margin, False otherwise.
        """
        if order.details.direction == Order.Direction.LONG:
            required_margin = cost * self.margin_ratio
        else:  # SHORT
            required_margin = (
                cost * self.margin_ratio * ExtendedDecimal("2")
            )  # Higher margin for short selling

        return self.buying_power >= required_margin

    def _update_margin(self, order: Order, cost: ExtendedDecimal) -> None:
        """
        Update margin and buying power after executing an order.

        This method calculates the required margin based on the order type and updates
        the margin_used attribute. It also recalculates the buying power based on
        the new equity and margin situation.

        Args:
            order (Order): The executed order.
            cost (ExtendedDecimal): The cost of the order.
        """
        if order.details.direction == Order.Direction.LONG:
            self.margin_used += cost * self.margin_ratio
        else:  # SHORT
            self.margin_used += cost * self.margin_ratio * ExtendedDecimal("2")

        self._update_buying_power()

    def _update_buying_power(self) -> None:
        """
        Update the buying power based on current equity and margin used.

        This method recalculates the available buying power considering the current
        equity, margin used, and margin ratio.
        """
        self.buying_power = (
            self.calculate_equity() - self.margin_used
        ) / self.margin_ratio

    def _check_margin_call(self) -> bool:
        """
        Check if a margin call should be triggered.

        Returns:
            bool: True if a margin call should be triggered, False otherwise.
        """
        equity = self.calculate_equity()
        if self.margin_used:
            if equity / self.margin_used < self.margin_call_threshold:
                logger_main.info("Margin call triggered!")
                return True
        return False

    def _handle_margin_call(self) -> None:
        """
        Handle a margin call by closing positions until margin requirements are met.

        This method closes the largest trades until the margin call is satisfied,
        and adds all closed trades to the updated_trades list for strategy notification.
        """
        while self._check_margin_call() and self.open_trades:
            largest_trade = max(
                (trade for trades in self.open_trades.values() for trade in trades),
                key=lambda t: abs(t.current_size * t.entry_price),
            )
            current_price = self.current_market_data[largest_trade.ticker][
                min(self.engine._dataview.data[largest_trade.ticker].keys())
            ]
            self.close_trade(largest_trade, current_price)

    # endregion

    # region Portfolio Analysis and Reporting

    def get_metrics_data(self) -> pd.DataFrame:
        """
        Retrieve the complete metrics DataFrame.

        This method returns the entire metrics DataFrame, including all calculated
        metrics and symbol-specific data.

        Returns:
            pd.DataFrame: The complete metrics DataFrame.
        """
        return self.metrics

    def get_current_equity(self) -> ExtendedDecimal:
        """
        Get the current total equity of the portfolio.

        Returns:
            ExtendedDecimal: The current equity value.
        """
        if len(self.metrics) > 0:
            return ExtendedDecimal(str(self.metrics.iloc[-1]["equity"]))
        return self.initial_capital

    def get_current_cash(self) -> ExtendedDecimal:
        """
        Get the current cash balance of the portfolio.

        Returns:
            ExtendedDecimal: The current cash balance.
        """
        return self.cash

    def get_open_positions(self) -> Dict[str, ExtendedDecimal]:
        """
        Get the current open positions in the portfolio.

        Returns:
            Dict[str, ExtendedDecimal]: A dictionary mapping symbols to their position sizes.
        """
        return self.positions

    def get_total_pnl(self) -> ExtendedDecimal:
        """
        Get the total profit and loss (PnL) of the portfolio.

        This includes both realized and unrealized PnL.

        Returns:
            ExtendedDecimal: The total PnL.
        """
        if len(self.metrics) > 0:
            return ExtendedDecimal(
                str(
                    self.metrics.iloc[-1]["open_pnl"]
                    + self.metrics.iloc[-1]["closed_pnl"]
                )
            )
        return ExtendedDecimal("0")

    # endregion

    # region Utility Methods

    def calculate_equity(self) -> ExtendedDecimal:
        """
        Calculate the total portfolio equity.

        This method computes the total equity by summing the cash balance,
        the value of long positions (asset_value), and subtracting the value
        of short positions (liabilities).

        Returns:
            ExtendedDecimal: The total portfolio equity.
        """
        return self.cash + self.long_position_value - self.short_position_value

    def get_account_value(self) -> ExtendedDecimal:
        """
        Get the current total account value (equity).

        Returns:
            ExtendedDecimal: The current account value.
        """
        return self.calculate_equity()

    def get_position_size(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            ExtendedDecimal: The current position size. Positive for long positions, negative for short positions.
        """
        return self.positions.get(symbol, ExtendedDecimal("0"))

    def get_available_margin(self) -> ExtendedDecimal:
        """
        Get the available margin for new trades.

        Returns:
            ExtendedDecimal: The available margin.
        """
        return self.buying_power

    def get_open_trades(self) -> List[Trade]:
        """
        Get all open trades.

        Returns:
            List[Trade]: A list of all open trades.
        """
        return [trade for trades in self.open_trades.values() for trade in trades]

    def get_closed_trades(self) -> List[Trade]:
        """
        Get all closed trades.

        Returns:
            List[Trade]: A list of all closed trades.
        """
        return self.closed_trades

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.
        """
        return [
            trade for trade in self.closed_trades if trade.strategy_id == strategy_id
        ] + [
            trade
            for trades in self.open_trades.values()
            for trade in trades
            if trade.strategy_id == strategy_id
        ]

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get the current state of the portfolio.

        Returns:
            Dict[str, Any]: A dictionary containing the current portfolio state,
                            including cash, equity, positions, trades, and other relevant metrics.
        """
        realized_pnl, unrealized_pnl = self._calculate_pnl()
        return {
            "cash": self.cash,
            "equity": self.calculate_equity(),
            "long_position_value": self.long_position_value,
            "short_position_value": self.short_position_value,
            "open_trades": {
                symbol: [trade.to_dict() for trade in trades]
                for symbol, trades in self.open_trades.items()
            },
            "closed_trades_count": len(self.closed_trades),
            "pending_orders": [order.to_dict() for order in self.pending_orders],
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "margin_used": self.margin_used,
            "buying_power": self.buying_power,
            "total_trades": self.trade_count,
            "positions": self.positions,
        }

    def _generate_order_id(self) -> int:
        """
        Generate a unique order ID.

        Returns:
            int: A unique order ID.
        """
        return hash(f"order_{datetime.now().timestamp()}_{len(self.pending_orders)}")

    def _create_dummy_bar(self, price: ExtendedDecimal, trade: Trade) -> Bar:
        """
        Create a dummy Bar object for trade closing operations.

        Args:
            price (ExtendedDecimal): The price to use for the dummy bar.
            trade (Trade): The trade associated with this bar, used for timeframe and ticker information.

        Returns:
            Bar: A dummy Bar object with the given price and current timestamp.
        """
        return Bar(
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0,
            timestamp=self.engine._current_timestamp,
            timeframe=trade.entry_bar.timeframe,
            ticker=trade.ticker,
        )

    # endregion
