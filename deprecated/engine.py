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
    1. Data Management:
       - Handles the loading of financial data from various sources (e.g., CSV files, APIs, databases)
       - Preprocesses and aligns data across different timeframes and symbols
       - Efficiently stores data using both pandas DataFrames (for flexibility) and optimized NumPy arrays (for performance)
       - Supports multiple timeframes and symbols simultaneously
       - Provides methods for data resampling and interpolation
       - Ensures data integrity and consistency throughout the backtesting process

    2. Strategy Execution:
       - Coordinates the application of multiple trading strategies concurrently
       - Manages strategy initialization, updating, and finalization
       - Handles strategy-specific parameters and configurations
       - Provides mechanisms for strategies to access market data and place orders
       - Supports various types of trading strategies (e.g., trend-following, mean-reversion, machine learning-based)
       - Allows for strategy performance tracking and dynamic parameter updates

    3. Portfolio Management:
       - Interfaces with the Portfolio class to manage trades and positions
       - Tracks open and closed positions across multiple symbols
       - Calculates and updates portfolio value, cash balance, and margin requirements
       - Implements risk management features such as position sizing and stop-loss orders
       - Handles order execution, including various order types (market, limit, stop, etc.)
       - Simulates realistic trading conditions, including slippage and transaction costs

    4. Backtesting Control:
       - Manages the overall flow of the backtesting process
       - Handles initialization of all components (data, strategies, portfolio)
       - Executes the main event loop, processing market events chronologically
       - Coordinates interactions between strategies, portfolio, and market data
       - Implements termination conditions (e.g., end date, minimum equity)
       - Ensures proper finalization of all components at the end of the backtest

    5. Performance Analysis:
       - Generates comprehensive performance reports upon backtest completion
       - Calculates key performance metrics (e.g., total return, Sharpe ratio, maximum drawdown)
       - Produces trade-by-trade analysis and summary statistics
       - Creates visualizations of performance, including equity curves and drawdown charts
       - Supports comparison of multiple strategy variants or parameter sets
       - Provides tools for analyzing strategy behavior and decision-making process

    Attributes:
        _dataview (DataView):
            Manages and aligns financial data across different timeframes and symbols.
            Responsible for data storage, retrieval, and preprocessing.

        _optimized_dataview (DataViewNumpy):
            Optimized version of the data view using NumPy arrays for faster data access.
            Provides high-performance data retrieval for intensive backtesting operations.

        portfolio (Portfolio):
            Manages trade execution, position tracking, and performance measurement.
            Handles all aspects of portfolio management, including cash balance and risk metrics.

        _strategies (Dict[str, Strategy]):
            Dictionary of Strategy objects, keyed by strategy ID.
            Stores all active strategies in the backtest, allowing for multi-strategy simulations.

        _current_timestamp (pd.Timestamp):
            Current timestamp being processed in the backtest.
            Used to synchronize all components and ensure chronological processing of events.

        _is_running (bool):
            Flag indicating whether a backtest is currently in progress.
            Helps prevent concurrent backtest runs and manage the engine's state.

        _config (Dict[str, Any]):
            Configuration dictionary for backtest parameters.
            Stores global settings such as initial capital, commission rates, and risk limits.

        _strategy_timeframes (Dict[str, Timeframe]):
            Primary timeframes for each strategy, keyed by strategy ID.
            Allows each strategy to operate on its preferred data frequency.

    Usage:
        The Engine class is the main entry point for setting up and running a backtest.
        Typical usage involves the following steps:
        1. Create an Engine instance
        2. Load data using add_data() or a DataLoader
        3. Add one or more strategies using add_strategy()
        4. Set configuration parameters with set_config()
        5. Run the backtest using the run() method
        6. Analyze the results using the generated performance metrics and visualizations

    Example:
        # Create an Engine instance
        engine = Engine()

        # Load data
        data_loader = YFDataloader(symbols=['AAPL', 'GOOGL'], timeframe='1d', start_date='2020-01-01', end_date='2021-12-31')
        engine.add_data(data_loader)

        # Create and add a strategy
        ma_cross_strategy = MovingAverageCrossoverStrategy(short_window=50, long_window=200)
        engine.add_strategy(ma_cross_strategy)

        # Set configuration
        engine.set_config({
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005
        })

        # Run the backtest
        results = engine.run()

        # Analyze results
        print(results['performance_metrics'])
        plot_equity_curve(results['equity_curve'])

    Notes:
        - The Engine class is designed to be flexible and extensible. Users can easily add
          custom data loaders, strategies, and performance metrics.
        - For large datasets or computationally intensive strategies, consider using the
          optimized NumPy-based data view for improved performance.
        - The Engine supports multi-threaded execution of strategies, but users should be
          aware of potential race conditions when implementing custom strategies.
        - While the Engine simulates many aspects of real trading, including slippage and
          commission costs, users should always validate backtest results with out-of-sample
          data and consider real-world factors not captured in the simulation.
        - The Engine's modular design allows for easy integration with external data sources,
          risk management tools, and reporting systems.
        - Regular logging and checkpointing during long backtests is recommended to track
          progress and allow for resuming in case of interruptions.

    This Engine class provides a sophisticated framework for conducting realistic and flexible
    backtests of trading strategies. It combines efficient data management, robust strategy
    execution, and comprehensive performance analysis to enable thorough evaluation and
    optimization of trading algorithms across various market conditions and instruments.
    """

    # region --- Initialization and Configuration

    def __init__(self):
        """Initialize the Engine instance."""
        self._dataview: DataView = DataView()
        self._optimized_dataview: Optional[DataViewNumpy] = None
        self.portfolio: Portfolio = Portfolio()
        self._strategies: Dict[str, Strategy] = {}
        self._current_timestamp: Optional[pd.Timestamp] = None
        self._is_running: bool = False
        self._config: Dict[str, Any] = {}
        self._strategy_timeframes: Dict[str, Timeframe] = {}

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the backtest.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self._config = config
        self.portfolio.set_config(config)  # Pass configuration to portfolio
        logger_main.log_and_print("Backtest configuration set.", level="info")

    def reset(self) -> None:
        """
        Reset the engine to its initial state.
        """
        self._dataview = DataView()
        self._optimized_dataview = None
        self.portfolio = Portfolio()
        self._strategies = {}
        self._current_timestamp = None
        self._is_running = False
        self._config = {}
        self._strategy_timeframes = {}
        logger_main.log_and_print("Engine reset to initial state.", level="info")

    def validate_data(self) -> bool:
        """
        Validate the loaded data to ensure it's sufficient for backtesting.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if not self._dataview.has_data:
            logger_main.log_and_print("No data has been loaded.", level="warning")
            return False

        if len(self._dataview.symbols) == 0:
            logger_main.log_and_print(
                "No symbols found in the loaded data.", level="warning"
            )
            return False

        if len(self._dataview.timeframes) == 0:
            logger_main.log_and_print(
                "No timeframes found in the loaded data.", level="warning"
            )
            return False

        date_range = self._dataview.get_data_range()
        if date_range[0] == date_range[1]:
            logger_main.log_and_print(
                "Insufficient data: only one data point available.", level="warning"
            )
            return False

        logger_main.log_and_print("Data validation passed.", level="info")
        return True

    # endregion

    # region --- Data Management
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
            f"Added data for {len(dataloader.dataframes)} symbols.", level="info"
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

        self.add_data(dataloader)
        logger_main.log_and_print(
            f"Resampled and added data for {len(dataloader.dataframes)} symbols to {timeframe}.",
            level="info",
        )

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
        try:
            return Bar(
                open=Decimal(str(ohlcv_data[0])),
                high=Decimal(str(ohlcv_data[1])),
                low=Decimal(str(ohlcv_data[2])),
                close=Decimal(str(ohlcv_data[3])),
                volume=int(ohlcv_data[4]),
                timestamp=self._current_timestamp,
                timeframe=timeframe,
                ticker=symbol,
            )
        except IndexError as e:
            logger_main.error(f"Error creating bar for {symbol} at {timeframe}: {e}")
            logger_main.error(f"OHLCV data: {ohlcv_data}")
            raise ValueError(f"Invalid OHLCV data for {symbol} at {timeframe}")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dict[str, Any]: A dictionary containing information about the loaded data.
        """
        return {
            "symbols": self._dataview.symbols,
            "timeframes": [str(tf) for tf in self._dataview.timeframes],
            "date_range": self._dataview.get_data_range(),
            "total_bars": sum(len(data) for data in self._dataview.data.values()),
        }

    def get_market_data(
        self, symbol: str, timeframe: Timeframe, n_bars: int = 1
    ) -> pd.DataFrame:
        """
        Get recent market data for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to get data for.
            timeframe (Timeframe): The timeframe of the data.
            n_bars (int): The number of recent bars to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the requested market data.
        """
        if self._optimized_dataview is None:
            logger_main.log_and_print(
                "Optimized dataview is not initialized.", level="warning"
            )
            return pd.DataFrame()

        end_idx = self._optimized_dataview.timestamp_to_index[self._current_timestamp]
        start_idx = max(0, end_idx - n_bars + 1)

        symbol_idx = self._optimized_dataview.symbol_to_index[symbol]
        timeframe_idx = self._optimized_dataview.timeframe_to_index[timeframe]

        data = self._optimized_dataview.data_array[
            symbol_idx, timeframe_idx, start_idx : end_idx + 1
        ]

        df = pd.DataFrame(
            data, columns=["open", "high", "low", "close", "volume", "is_original"]
        )
        df.index = self._optimized_dataview.master_timeline[start_idx : end_idx + 1]
        return df

    # endregion

    # region --- Strategy Management
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
        strategy.set_engine(self)
        self._strategies[strategy._id] = strategy
        if primary_timeframe is not None:
            self._strategy_timeframes[strategy._id] = primary_timeframe
        logger_main.info(
            f"Added strategy: {strategy.__class__.__name__} with primary timeframe {strategy.primary_timeframe}"
        )

    def remove_strategy(self, strategy_id: str) -> None:
        """
        Remove a trading strategy from the engine.

        Args:
            strategy_id (str): The ID of the strategy to be removed.
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger_main.log_and_print(
                f"Removed strategy with ID: {strategy_id}", level="info"
            )
        else:
            logger_main.log_and_print(
                f"Strategy not found with ID: {strategy_id}", level="warning"
            )

    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by its ID.

        Args:
            strategy_id (str): The ID of the strategy to retrieve.

        Returns:
            Optional[Strategy]: The strategy with the given ID, or None if not found.
        """
        return self._strategies.get(strategy_id)

    def update_strategy_parameters(
        self, strategy_id: str, new_parameters: Dict[str, Any]
    ) -> None:
        """
        Update the parameters of a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy to update.
            new_parameters (Dict[str, Any]): A dictionary of new parameter values.
        """
        strategy = self.get_strategy_by_id(strategy_id)
        if strategy:
            strategy.parameters = new_parameters
            logger_main.log_and_print(
                f"Updated parameters for strategy {strategy_id}", level="info"
            )
        else:
            logger_main.log_and_print(
                f"Strategy with ID {strategy_id} not found.", level="warning"
            )

    def log_strategy_activity(self, strategy_id: str, message: str) -> None:
        """
        Log activity for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.
            message (str): The message to log.
        """
        logger_main.log_and_print(f"Strategy {strategy_id}: {message}", level="info")

    # endregion

    # region --- Backtesting Execution

    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest and return the results.

        Returns:
            Dict[str, Any]: The final backtest results.

        Raises:
            ValueError: If data validation fails.
            Exception: For any errors occurring during the backtest.
        """
        if self._is_running:
            logger_main.log_and_print("Backtest is already running.", level="warning")
            return {}
        try:
            self._init_dataview_numpy()

            if not self.validate_data():
                logger_main.log_and_raise(
                    ValueError("Data validation failed. Cannot start backtest.")
                )

            self._is_running = True
            self._initialize_backtest()
            logger_main.log_and_print("Starting backtest...", level="info")

            for timestamp, data_point in self._optimized_dataview:
                self._current_timestamp = timestamp
                self._process_timestamp(timestamp, data_point)

                if self._check_termination_condition():
                    break

            self._finalize_backtest()
        except Exception as e:
            logger_main.log_and_raise(Exception(f"Error during backtest: {str(e)}"))
        finally:
            self._is_running = False

        logger_main.log_and_print("Backtest completed.", level="info")
        return self._generate_results()

    def _initialize_backtest(self) -> None:
        """Initialize all components for the backtest."""
        default_timeframe = min(self._optimized_dataview.timeframes)
        for strategy_id, strategy in self._strategies.items():
            primary_timeframe = self._strategy_timeframes.get(
                strategy_id, default_timeframe
            )
            strategy.initialize(
                self._optimized_dataview.symbols,
                self._optimized_dataview.timeframes,
                primary_timeframe,
            )
        logger_main.log_and_print("Backtest initialized.", level="info")

    def _init_dataview_numpy(self) -> None:
        """Initialize the optimized NumPy-based DataView."""
        self._dataview.align_all_data()
        self._optimized_dataview = DataViewNumpy(self._dataview)
        logger_main.log_and_print(
            "Initialized optimized NumPy-based DataView.", level="info"
        )

    def _process_timestamp(
        self,
        timestamp: pd.Timestamp,
        data_point: Dict[str, Dict[Timeframe, np.ndarray]],
    ) -> None:
        """
        Process a single timestamp across all symbols and timeframes.

        Args:
            timestamp (pd.Timestamp): The current timestamp being processed.
            data_point (Dict[str, Dict[Timeframe, np.ndarray]]): Market data for the current timestamp.
        """
        for strategy_id, strategy in self._strategies.items():
            for symbol in self._optimized_dataview.symbols:
                if self._optimized_dataview.is_original_data_point(
                    symbol, strategy.primary_timeframe, timestamp
                ):
                    ohlcv_data = data_point[symbol][strategy.primary_timeframe]
                    bar = self._create_bar(
                        symbol, strategy.primary_timeframe, ohlcv_data
                    )
                    strategy.process_bar(bar)

        # Process order and trade updates
        for order in self.portfolio.get_updated_orders():
            self._process_order_update(order)
        for trade in self.portfolio.get_updated_trades():
            self._process_trade_update(trade)

    def _check_termination_condition(self) -> bool:
        """
        Check if the backtest should be terminated based on certain conditions.

        Returns:
            bool: True if the backtest should be terminated, False otherwise.
        """
        if self.portfolio.calculate_equity() < self._config.get("min_equity", 0):
            logger_main.log_and_print(
                "Portfolio value dropped below minimum equity. Terminating backtest.",
                level="warning",
            )
            return True
        return False

    def _finalize_backtest(self) -> None:
        """Perform final operations after the backtest is complete."""
        self.portfolio.close_all_trades(
            self._current_timestamp, self._optimized_dataview
        )
        logger_main.log_and_print(
            "Finalized backtest. Closed all remaining trades.", level="info"
        )

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate and return the final backtest results.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results and performance metrics.
        """
        return {
            "performance_metrics": self.portfolio.get_performance_metrics(),
            "trade_history": self.portfolio.get_trade_history(),
            "equity_curve": self.portfolio.get_equity_curve(),
        }

    # endregion

    # region --- Order and Trade Management

    def create_order(
        self,
        strategy_id: str,
        symbol: str,
        direction: Order.Direction,
        size: float,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Order]:
        """
        Create an order based on a strategy's request.

        Args:
            strategy_id (str): The ID of the strategy initiating the order.
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
        order = self.portfolio.create_order(
            symbol,
            direction,
            size,
            order_type,
            price,
            strategy_id=strategy_id,
            **kwargs,
        )

        return order

    def cancel_order(self, strategy_id: str, order: Order) -> bool:
        """
        Cancel an existing order.

        Args:
            strategy_id (str): The ID of the strategy requesting the cancellation.
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order.strategy_id != strategy_id:
            logger_main.log_and_print(
                f"Strategy {strategy_id} attempted to cancel an order it didn't create.",
                level="warning",
            )
            return False

        success = self.portfolio.cancel_order(order)
        return success

    def close_positions(self, strategy_id: str, symbol: Optional[str] = None) -> bool:
        """
        Close all positions for a strategy, or for a specific symbol if provided.

        Args:
            strategy_id (str): The ID of the strategy requesting to close positions.
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions for the strategy.

        Returns:
            bool: True if the closing operation was successful, False otherwise.
        """
        success = self.portfolio.close_positions(strategy_id=strategy_id, symbol=symbol)
        return success

    def _process_order_update(self, order: Order) -> None:
        """
        Process updates to an order and notify the relevant strategy.

        Args:
            order (Order): The updated order.
        """
        strategy = self.get_strategy_by_id(order.strategy_id)
        if strategy:
            strategy.on_order_update(order)

    def _process_trade_update(self, trade: Trade) -> None:
        """
        Process updates to a trade and notify the relevant strategy.

        Args:
            trade (Trade): The updated trade.
        """
        strategy = self.get_strategy_by_id(trade.strategy_id)
        if strategy:
            strategy.on_trade_update(trade)

    # endregion

    # region --- Portfolio and Account Information
    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            List[Trade]: A list of Trade objects associated with the strategy.
        """
        return self.portfolio.get_trades_for_strategy(strategy_id)

    def get_account_value(self) -> Decimal:
        """
        Get the current total account value (equity).

        Returns:
            Decimal: The current account value.
        """
        return self.portfolio.get_account_value()

    def get_position_size(self, symbol: str) -> Decimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            Decimal: The current position size. Positive for long positions, negative for short positions.
        """
        return self.portfolio.get_position_size(symbol)

    def get_available_margin(self) -> Decimal:
        """
        Get the available margin for new trades.

        Returns:
            Decimal: The available margin.
        """
        return self.portfolio.get_available_margin()

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the backtest.
        """
        return {
            "current_timestamp": self._current_timestamp,
            "portfolio_state": self.portfolio.get_portfolio_state(),
            "is_running": self._is_running,
            "strategies": list(self._strategies.keys()),
        }

    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the performance metrics for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics for the strategy.
        """
        strategy = self.get_strategy_by_id(strategy_id)
        if not strategy:
            logger_main.log_and_print(
                f"Strategy with ID {strategy_id} not found.", level="warning"
            )
            return {}

        trades = self.get_trades_for_strategy(strategy_id)
        # Calculate and return performance metrics
        return {
            "total_trades": len(trades),
            "winning_trades": sum(1 for trade in trades if trade.metrics.pnl > 0),
            "losing_trades": sum(1 for trade in trades if trade.metrics.pnl < 0),
            "total_pnl": sum(trade.metrics.pnl for trade in trades),
            # Add more metrics as needed
        }

    # endregion
