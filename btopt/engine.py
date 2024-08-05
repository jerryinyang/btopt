from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .data.bar import Bar
from .data.dataloader import BaseDataLoader
from .data.dataview import DataView, DataViewNumpy
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order
from .portfolio import Portfolio
from .strategy.strategy import Strategy
from .trade import Trade


class Engine:
    """
    A comprehensive backtesting engine for financial trading strategies.

    This class serves as the central coordinator for the entire backtesting process,
    orchestrating data management, strategy execution, and portfolio operations.
    It provides a robust framework for simulating and evaluating trading strategies
    across multiple financial instruments and timeframes.

    Key Responsibilities:
    1. Data Management: Handles the loading, preprocessing, and efficient storage
       of financial data using both pandas DataFrames and optimized NumPy arrays.
    2. Strategy Execution: Coordinates the application of multiple trading strategies
       across various symbols and timeframes.
    3. Portfolio Management: Interfaces with the Portfolio class to manage trades,
       track performance, and calculate risk metrics.
    4. Backtesting Control: Manages the overall flow of the backtesting process,
       including initialization, main loop execution, and finalization.
    5. Performance Analysis: Generates comprehensive performance reports and metrics
       upon completion of the backtest.

    The Engine class is designed to be flexible and extensible, allowing for easy
    integration of new data sources, trading strategies, and performance metrics.

    Attributes:
        _dataview (DataView):
            An instance of the DataView class that stores and manages financial data.
            This object handles data alignment across different timeframes and symbols,
            ensuring consistent and synchronized data access throughout the backtest.

        _optimized_dataview (DataViewNumpy):
            An optimized version of the data view using NumPy arrays for improved
            performance. This is particularly useful for high-frequency data or
            when dealing with large datasets.

        _portfolio (Portfolio):
            An instance of the Portfolio class that manages all aspects of trade
            execution, position tracking, and performance measurement. This includes
            handling of orders, calculation of returns, and risk management.

        _strategies (List[Strategy]):
            A list of Strategy objects representing the trading strategies to be
            tested. Each strategy can generate trading signals for multiple symbols
            and timeframes.

        _current_timestamp (pd.Timestamp):
            The current timestamp being processed in the backtest. This is updated
            at each iteration of the main backtest loop and is used to synchronize
            data, strategy signals, and portfolio updates.

        _is_running (bool):
            A flag indicating whether a backtest is currently in progress. This is
            used to prevent multiple concurrent backtests and to manage the backtest
            state.

        _config (Dict[str, Any]):
            A configuration dictionary containing various parameters for the backtest.
            This may include settings such as initial capital, commission rates,
            slippage models, risk limits, and other customizable aspects of the
            simulation.

    Methods:
        The Engine class provides a wide range of methods for setting up, running,
        and analyzing backtests. These include methods for data loading and
        preprocessing, strategy management, backtest execution control, and
        results generation. Detailed documentation for each method is provided
        in their respective docstrings.

    Usage:
        The Engine class is typically used as the main entry point for setting up
        and running a backtest. A typical workflow might involve:
        1. Instantiating the Engine
        2. Loading and preprocessing data
        3. Adding one or more trading strategies
        4. Configuring backtest parameters
        5. Running the backtest
        6. Analyzing and visualizing the results

    Example:
        engine = Engine()
        engine.add_data(data_loader)
        engine.add_strategy(MyStrategy())
        engine.set_config({"initial_capital": 100000, "commission_rate": 0.001})
        results = engine.run()
        engine.plot_results()

    Note:
        The Engine class is designed to be thread-safe and can be used in
        multi-threaded environments, although care should be taken when
        accessing shared resources.
    """

    def __init__(self):
        """Initialize the Engine instance."""
        self._dataview = DataView()
        self._optimized_dataview = None
        self._portfolio = Portfolio()
        self._strategies: List[Strategy] = []
        self._current_timestamp = None
        self._is_running = False
        self._config = {}

    # region Backtest Execution
    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest and return the results.

        This method orchestrates the entire backtesting process, including initialization,
        data processing, and result generation.

        Returns:
            Dict[str, Any]: The final backtest results.

        Raises:
            ValueError: If data validation fails.
            Exception: For any errors occurring during the backtest.
        """
        if self._is_running:
            logger_main.warning("Backtest is already running.")
            return {}

        if not self.validate_data():
            logger_main.log_and_raise(
                ValueError("Data validation failed. Cannot start backtest.")
            )

        self._is_running = True
        logger_main.info("Starting backtest...")

        try:
            self._initialize_backtest()

            for timestamp, data_point in self._optimized_dataview:
                self._current_timestamp = timestamp
                self._process_timestamp(timestamp)

                if self._check_termination_condition():
                    break

            self._finalize_backtest()
        except Exception as e:
            logger_main.error(f"Error during backtest: {str(e)}")
            raise
        finally:
            self._is_running = False

        logger_main.info("Backtest completed.")
        return self._generate_results()

    def _initialize_backtest(self):
        """
        Initialize all components for the backtest.

        This method prepares the optimized dataview and initializes all strategies.
        """
        default_timeframe = min(self._optimized_dataview.timeframes)
        for strategy in self._strategies:
            primary_timeframe = self._strategy_timeframes.get(
                strategy, default_timeframe
            )
            strategy.initialize(
                self._optimized_dataview.symbols,
                self._optimized_dataview.timeframes,
                primary_timeframe,
            )
        logger_main.info("Backtest initialized.")

    def _process_timestamp(self, timestamp: pd.Timestamp):
        """
        Process a single timestamp across all symbols and timeframes.

        This method generates market data, creates signals from strategies,
        and passes the information to the portfolio for processing.

        Args:
            timestamp (pd.Timestamp): The current timestamp being processed.
        """
        market_data = self._get_market_data(timestamp)
        signals = []

        for strategy in self._strategies:
            strategy_bars = {}
            for symbol in self._optimized_dataview.symbols:
                if self._optimized_dataview.is_original_data_point(
                    symbol, strategy.primary_timeframe, timestamp
                ):
                    ohlcv_data = market_data[symbol][strategy.primary_timeframe]
                    strategy_bars[symbol] = self._create_bar(
                        symbol, strategy.primary_timeframe, ohlcv_data
                    )

            if strategy_bars:
                strategy.on_bar(timestamp, strategy_bars)
                for symbol, bar in strategy_bars.items():
                    signals.extend(
                        strategy.generate_signals(
                            symbol, strategy.primary_timeframe, timestamp, bar
                        )
                    )

        standardized_signals = self._standardize_signals(signals)
        self._portfolio.process_signals(standardized_signals, timestamp, market_data)

        # Process order and trade updates
        for order in self._portfolio.get_updated_orders():
            self._process_order_update(order)
        for trade in self._portfolio.get_updated_trades():
            self._process_trade_update(trade)

    def _check_termination_condition(self) -> bool:
        """
        Check if the backtest should be terminated based on certain conditions.

        Returns:
            bool: True if the backtest should be terminated, False otherwise.
        """
        # Implement your termination conditions here
        # For example, you might want to stop if the portfolio value drops below a certain threshold
        if self._portfolio.calculate_equity() < self._config.get("min_equity", 0):
            logger_main.warning(
                "Portfolio value dropped below minimum equity. Terminating backtest."
            )
            return True
        return False

    def _finalize_backtest(self):
        """
        Perform final operations after the backtest is complete.
        """
        self._portfolio.close_all_trades(
            self._current_timestamp, self._optimized_dataview
        )
        logger_main.info("Finalized backtest. Closed all remaining trades.")

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate and return the final backtest results.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results and performance metrics.
        """
        return {
            "performance_metrics": self._portfolio.get_performance_metrics(),
            "trade_history": self._portfolio.get_trade_history(),
            "equity_curve": self._portfolio.get_equity_curve(),
        }

    def register_strategy(self, strategy: Strategy) -> None:
        """
        Register a strategy with the engine and assign the engine reference to the strategy.

        Args:
            strategy (Strategy): The strategy to register.
        """
        strategy._engine = self
        self._strategies.append(strategy)
        logger_main.info(f"Registered strategy: {strategy.name} (ID: {strategy._id})")

    def create_order(
        self,
        strategy: Strategy,
        symbol: str,
        direction: Order.Direction,
        size: float,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Order]:
        """
        Create an order based on a strategy's request.

        Args:
            strategy (Strategy): The strategy initiating the order.
            symbol (str): The symbol to trade.
            direction (Order.Direction): The direction of the trade (LONG or SHORT).
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            **kwargs: Additional order parameters.

        Returns:
            Optional[Order]: The created Order object, or None if order creation failed.
        """
        order_type = (
            Order.ExecType.LIMIT if price is not None else Order.ExecType.MARKET
        )
        order = self._portfolio.create_order(
            symbol,
            direction,
            size,
            order_type,
            price,
            strategy_id=strategy._id,
            **kwargs,
        )

        if order:
            strategy.on_order(order)
        return order

    def cancel_order(self, strategy: Strategy, order: Order) -> bool:
        """
        Cancel an existing order.

        Args:
            strategy (Strategy): The strategy requesting the cancellation.
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order.strategy_id != strategy._id:
            logger_main.warning(
                f"Strategy {strategy.name} attempted to cancel an order it didn't create."
            )
            return False

        success = self._portfolio.cancel_order(order)
        if success:
            strategy.on_order(order)  # Notify the strategy of the order cancellation
        return success

    def close_positions(self, strategy: Strategy, symbol: Optional[str] = None) -> bool:
        """
        Close all positions for a strategy, or for a specific symbol if provided.

        Args:
            strategy (Strategy): The strategy requesting to close positions.
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions for the strategy.

        Returns:
            bool: True if the closing operation was successful, False otherwise.
        """
        success = self._portfolio.close_positions(
            strategy_id=strategy._id, symbol=symbol
        )
        if success:
            # Notify the strategy of closed positions
            closed_trades = self._portfolio.get_closed_trades(
                strategy_id=strategy._id, symbol=symbol
            )
            for trade in closed_trades:
                strategy.on_trade(trade)
        return success

    def _process_order_update(self, order: Order) -> None:
        """
        Process updates to an order and notify the relevant strategy.

        Args:
            order (Order): The updated order.
        """
        strategy = self._get_strategy_by_id(order.strategy_id)
        if strategy:
            strategy.on_order(order)

    def _process_trade_update(self, trade: Trade) -> None:
        """
        Process updates to a trade and notify the relevant strategy.

        Args:
            trade (Trade): The updated trade.
        """
        strategy = self._get_strategy_by_id(trade.strategy_id)
        if strategy:
            strategy.on_trade(trade)

    def _get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by its ID.

        Args:
            strategy_id (str): The ID of the strategy to retrieve.

        Returns:
            Optional[Strategy]: The strategy with the given ID, or None if not found.
        """
        for strategy in self._strategies:
            if strategy._id == strategy_id:
                return strategy
        return None

    # endregion

    # region Data Management
    def add_data(self, dataloader: BaseDataLoader) -> None:
        """
        Add data from a dataloader to the engine's DataView.

        Args:
            dataloader (BaseDataLoader): An instance of a dataloader containing financial data.

        Raises:
            ValueError: If the dataloader does not contain any data.
        """
        if not dataloader.has_data:
            logger_main.log_and_raise(
                ValueError("Dataloader does not contain any data.")
            )

        for ticker, data in dataloader.dataframes.items():
            self._dataview.add_data(
                symbol=ticker,
                timeframe=dataloader.timeframe,
                df=data,
            )
        logger_main.log_and_print(
            f"Added data for {len(dataloader.dataframes)} symbols."
        )

    def resample_data(
        self, dataloader: BaseDataLoader, timeframe: Union[str, Timeframe]
    ) -> None:
        """
        Resample data from a dataloader to a new timeframe and add it to the engine's DataView.

        Args:
            dataloader (BaseDataLoader): An instance of a dataloader containing financial data.
            timeframe (Union[str, Timeframe]): The target timeframe for resampling.

        Raises:
            ValueError: If the dataloader does not contain any data.
        """
        if not dataloader.has_data:
            logger_main.log_and_raise(
                ValueError("Dataloader does not contain any data.")
            )

        for ticker, data in dataloader.dataframes.items():
            self._dataview.resample_data(
                symbol=ticker,
                from_timeframe=dataloader.timeframe,
                to_timeframe=timeframe,
                df=data,
            )
        logger_main.log_and_print(
            f"Resampled data for {len(dataloader.dataframes)} symbols to {timeframe}."
        )

    def _init_dataview_numpy(self) -> None:
        """
        Initialize the optimized NumPy-based DataView.

        This method aligns all data in the DataView and creates a DataViewNumpy instance.
        """
        self._dataview.align_all_data()
        self._optimized_dataview = DataViewNumpy(self._dataview)
        logger_main.log_and_print("Initialized optimized NumPy-based DataView.")

    def _get_market_data(
        self, timestamp: pd.Timestamp
    ) -> Dict[str, Dict[Timeframe, np.ndarray]]:
        """
        Retrieve market data for all symbols and timeframes at a given timestamp.

        Args:
            timestamp (pd.Timestamp): The timestamp for which to retrieve data.

        Returns:
            Dict[str, Dict[Timeframe, np.ndarray]]: A nested dictionary containing
            market data for each symbol and timeframe.
        """
        market_data = {}
        for symbol in self._optimized_dataview.symbols:
            market_data[symbol] = {}
            for timeframe in self._optimized_dataview.timeframes:
                data = self._optimized_dataview.get_data_point_by_keys(
                    symbol, timeframe, timestamp
                )
                if data is not None:
                    market_data[symbol][timeframe] = data
        return market_data

    def _create_bar(
        self, symbol: str, timeframe: Timeframe, ohlcv_data: np.ndarray
    ) -> Bar:
        """
        Create a Bar object from OHLCV data.

        Args:
            symbol (str): The symbol for the bar.
            timeframe (Timeframe): The timeframe of the bar.
            ohlcv_data (np.ndarray): Array containing OHLCV data.

        Returns:
            Bar: A Bar object representing the market data.
        """
        return Bar(
            open=ohlcv_data[0],
            high=ohlcv_data[1],
            low=ohlcv_data[2],
            close=ohlcv_data[3],
            volume=ohlcv_data[4],
            timestamp=self._current_timestamp,
            timeframe=timeframe,
            ticker=symbol,
        )

    # endregion

    # region Strategy Management
    def add_strategy(
        self, strategy: Strategy, primary_timeframe: Optional[Timeframe] = None
    ) -> None:
        """
        Add a trading strategy to the engine.

        Args:
            strategy (Strategy): The strategy to be added.
            primary_timeframe (Optional[Timeframe]): The primary timeframe for strategy updates.
                If None, the lowest timeframe in the data will be used.
        """
        self._strategies.append(strategy)
        if primary_timeframe is not None:
            self._strategy_timeframes[strategy] = primary_timeframe
        logger_main.info(f"Added strategy: {strategy.__class__.__name__}")

    def remove_strategy(self, strategy: Strategy) -> None:
        """
        Remove a trading strategy from the engine.

        Args:
            strategy (Strategy): The strategy to be removed.
        """
        if strategy in self._strategies:
            self._strategies.remove(strategy)
            logger_main.info(f"Removed strategy: {strategy.__class__.__name__}")
        else:
            logger_main.warning(f"Strategy not found: {strategy.__class__.__name__}")

    def _generate_signals(
        self, symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp, bar: Bar
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals from all strategies for a given symbol and timeframe.

        Args:
            symbol (str): The symbol being processed.
            timeframe (Timeframe): The timeframe of the data point.
            timestamp (pd.Timestamp): The timestamp of the data point.
            bar (Bar): The price bar data.

        Returns:
            List[Dict[str, Any]]: A list of signal dictionaries generated by the strategies.
        """
        signals = []
        for strategy in self._strategies:
            strategy_signals = strategy.generate_signals(
                symbol, timeframe, timestamp, bar
            )
            if strategy_signals:
                signals.extend(strategy_signals)
        return signals

    def _standardize_signals(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Standardize the format of strategy signals.

        This method ensures all signals have a consistent format before being
        passed to the portfolio for processing.

        Args:
            signals (List[Dict[str, Any]]): The original signals generated by strategies.

        Returns:
            List[Dict[str, Any]]: A list of standardized signal dictionaries.
        """
        standardized_signals = []
        for signal in signals:
            standardized_signal = {
                "symbol": signal["symbol"],
                "timeframe": signal["timeframe"],
                "direction": signal["direction"],
                "size": signal["size"],
                "price": signal["price"],
                "order_type": signal["order_type"],
                "timestamp": signal["timestamp"],
            }
            # Add optional fields if present
            for field in [
                "expiry",
                "stoplimit_price",
                "parent_id",
                "exit_profit",
                "exit_loss",
                "exit_profit_percent",
                "exit_loss_percent",
                "trailing_percent",
            ]:
                if field in signal:
                    standardized_signal[field] = signal[field]
            standardized_signals.append(standardized_signal)
        return standardized_signals

    def get_trades_for_strategy(self, strategy: Strategy) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy (Strategy): The strategy object.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.
        """
        return self._portfolio.get_trades_for_strategy(strategy._id)

    def get_account_value(self) -> Decimal:
        """
        Get the current total account value (equity).

        Returns:
            Decimal: The current account value.
        """
        return self._portfolio.get_account_value()

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            Decimal: The current position size. Positive for long positions, negative for short positions.
        """
        return self._portfolio.get_position_size(symbol)

    def get_available_margin(self) -> Decimal:
        """
        Get the available margin for new trades.

        Returns:
            Decimal: The available margin.
        """
        return self._portfolio.get_available_margin()

    # endregion

    # region Configuration and Utility Methods
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the backtest.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self._config = config
        self._portfolio.set_config(config)  # Pass configuration to portfolio
        logger_main.info("Backtest configuration set.")

    def validate_data(self) -> bool:
        """
        Validate the loaded data to ensure it's sufficient for backtesting.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if not self._dataview.has_data:
            logger_main.warning("No data has been loaded.")
            return False

        if len(self._dataview.symbols) == 0:
            logger_main.warning("No symbols found in the loaded data.")
            return False

        if len(self._dataview.timeframes) == 0:
            logger_main.warning("No timeframes found in the loaded data.")
            return False

        date_range = self._dataview.get_data_range()
        if date_range[0] == date_range[1]:
            logger_main.warning("Insufficient data: only one data point available.")
            return False

        logger_main.info("Data validation passed.")
        return True

    def reset(self):
        """
        Reset the engine to its initial state.
        """
        self._dataview = DataView()
        self._optimized_dataview = None
        self._portfolio = Portfolio()
        self._strategies = []
        self._current_timestamp = None
        self._is_running = False
        self._config = {}
        logger_main.info("Engine reset to initial state.")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the backtest.
        """
        return {
            "current_timestamp": self._current_timestamp,
            "portfolio_state": self._portfolio.get_portfolio_state(),
            "is_running": self._is_running,
        }

    # endregion
