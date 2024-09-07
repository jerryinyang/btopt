import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from .data.bar import Bar
from .data.dataloader import BaseDataLoader
from .data.dataview import DataView
from .data.timeframe import Timeframe
from .order import Order
from .portfolio import Portfolio
from .reporter import Reporter
from .strategy import Strategy
from .trade import Trade
from .util.ext_decimal import ExtendedDecimal
from .util.log_config import clear_log_files, logger_main


class Engine:
    """
    A comprehensive backtesting engine for financial trading strategies.

    The Engine class serves as the central component of a backtesting system, coordinating
    data management, strategy execution, and portfolio operations to simulate trading
    strategies on historical market data. It provides a robust framework for developing,
    testing, and analyzing trading strategies across multiple assets and timeframes.

    Key Features:
    1. Data Management: Handles loading, preprocessing, and storage of market data
       across multiple symbols and timeframes.
    2. Strategy Integration: Supports multiple trading strategies, allowing for
       concurrent testing and comparison.
    3. Portfolio Management: Simulates a trading portfolio, including order execution,
       position tracking, and performance calculation.
    4. Flexible Backtesting: Provides methods for running backtests with customizable
       parameters and constraints.
    5. Performance Analysis: Offers tools for calculating and visualizing backtest
       results, including equity curves, drawdowns, and trade statistics.

    The Engine class is designed to be modular and extensible, allowing for easy
    integration of new data sources, trading strategies, and analysis tools.

    Attributes:
        _dataview (DataView): Manages market data storage and retrieval.
        portfolio (Portfolio): Handles portfolio management and trade execution.
        _strategies (Dict[str, Strategy]): Stores active trading strategies.
        _current_timestamp (Optional[pd.Timestamp]): Current timestamp in the backtest.
        _is_running (bool): Indicates if a backtest is currently in progress.
        _config (Dict[str, Any]): Stores backtest configuration settings.
        _strategy_timeframes (Dict[str, Timeframe]): Maps strategies to their primary timeframes.

    Usage:
        1. Initialize the Engine.
        2. Load market data using add_data() or resample_data().
        3. Add one or more trading strategies using add_strategy().
        4. Configure the backtest parameters with set_config().
        5. Run the backtest using the run() method.
        6. Analyze results using various getter methods and analysis tools.

    Example:
        engine = Engine()
        engine.add_data(YahooFinanceDataloader("AAPL", "1d", start_date="2020-01-01"))
        engine.add_strategy(SimpleMovingAverageCrossover(fast_period=10, slow_period=30))
        engine.set_config({"initial_capital": 100000, "commission": 0.001})
        results = engine.run()
        print(engine.calculate_performance_metrics())

    Note:
        This class is the core of the backtesting system and interacts closely with
        other components such as DataView, Portfolio, and individual Strategy instances.
        It's designed to be thread-safe for potential future extensions into parallel
        or distributed backtesting scenarios.
    """

    def __init__(self):
        self._dataview: DataView = DataView()
        self.portfolio: Optional[Portfolio] = None
        self._strategies: Dict[str, Strategy] = {}
        self._current_timestamp: Optional[pd.Timestamp] = None
        self._current_market_data: Optional[Dict[str, Dict[Timeframe, Bar]]] = None
        self._is_running: bool = False
        self._config: Dict[str, Any] = {}
        self._strategy_timeframes: Dict[str, List[Timeframe]] = {}
        self.reporter: Optional[Reporter] = None

    # region Initialization and Configuration

    @property
    def default_timeframe(self) -> Timeframe:
        """
        Get the default (lowest) timeframe available in the DataView.

        Returns:
            Timeframe: The lowest timeframe available in the data.

        Raises:
            ValueError: If no data has been added to the DataView yet.
        """
        if self._dataview.lowest_timeframe is None:
            logger_main.log_and_raise(
                ValueError("No data has been added to the DataView yet.")
            )
        return self._dataview.lowest_timeframe

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the backtest.

        This method stores the configuration to be used when initializing the Portfolio
        during the backtest run.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self._config = config
        logger_main.info(
            "Configuration set. Portfolio will be initialized during backtest run."
        )

    def reset(self) -> None:
        """
        Reset the engine to its initial state.

        This method clears all data, strategies, and configurations, preparing the engine for a new backtest.
        """
        self._dataview = DataView()
        self.portfolio = Portfolio()
        self._strategies = {}
        self._current_timestamp = None
        self._is_running = False
        self._config = {}
        self._strategy_timeframes = {}
        logger_main.info("Engine reset to initial state.")

    def validate_data(self) -> bool:
        """
        Validate the loaded data to ensure it's sufficient for backtesting.

        Returns:
            bool: True if the data is valid and sufficient, False otherwise.
        """
        if not self._dataview.has_data:
            logger_main.info("No data has been loaded.")
            return False

        if len(self._dataview.symbols) == 0:
            logger_main.info("No symbols found in the loaded data.")
            return False

        if len(self._dataview.timeframes) == 0:
            logger_main.info("No timeframes found in the loaded data.")
            return False

        date_range = self._dataview.get_data_range()
        if date_range[0] == date_range[1]:
            logger_main.info("Insufficient data: only one data point available.")
            return False

        logger_main.info("Data validation passed.")
        return True

    # endregion

    # region Data Management

    def add_data(self, dataloader: BaseDataLoader) -> None:
        """
        Add data from a dataloader to the engine's DataView.

        Args:
            dataloader (BaseDataLoader): The dataloader containing market data to be added.

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
        logger_main.info(f"Added data for {len(dataloader.dataframes)} symbols.")

    def resample_data(
        self, dataloader: BaseDataLoader, timeframe: Union[str, Timeframe]
    ) -> None:
        """
        Resample data from a dataloader to a new timeframe and add it to the engine's DataView.

        Args:
            dataloader (BaseDataLoader): The dataloader containing market data to be resampled.
            timeframe (Union[str, Timeframe]): The target timeframe for resampling.

        Raises:
            ValueError: If the dataloader does not contain any data.
        """
        if not dataloader.has_data:
            logger_main.log_and_raise(
                ValueError("Dataloader does not contain any data.")
            )

        def resampling_modifier(
            dataframes: Dict[str, pd.DataFrame],
        ) -> Dict[str, pd.DataFrame]:
            resampled_dataframes = {}
            for ticker, data in dataframes.items():
                resampled_df = self._dataview.resample_data(
                    symbol=ticker,
                    from_timeframe=dataloader.timeframe,
                    to_timeframe=timeframe,
                    df=data,
                )
                resampled_dataframes[ticker] = resampled_df
            return resampled_dataframes

        resampled_dataloader = dataloader.create_modified(resampling_modifier)
        resampled_dataloader.timeframe = (
            timeframe if isinstance(timeframe, Timeframe) else Timeframe(timeframe)
        )

        # Add the resampled dataloader to the engine
        self.add_data(resampled_dataloader)

        logger_main.info(
            f"Resampled data for {len(resampled_dataloader.dataframes)} symbols to {timeframe}.",
        )

    def _create_bar(self, symbol: str, timeframe: Timeframe, data: pd.Series) -> Bar:
        """
        Create a Bar object from OHLCV data.

        Args:
            symbol (str): The symbol for which the bar is being created.
            timeframe (Timeframe): The timeframe of the bar.
            data (pd.Series): The OHLCV data for the bar.

        Returns:
            Bar: A Bar object representing the market data.
        """
        return Bar(
            open=ExtendedDecimal(str(data["open"])),
            high=ExtendedDecimal(str(data["high"])),
            low=ExtendedDecimal(str(data["low"])),
            close=ExtendedDecimal(str(data["close"])),
            volume=(data["volume"]),
            timestamp=self._current_timestamp,
            timeframe=timeframe,
            ticker=symbol,
        )

    def _create_bars_data(
        self,
        datapoints: Dict[str, Dict[Timeframe, pd.Series]],
    ) -> Dict[str, Dict[Timeframe, Bar]]:
        """
        Convert the data points into bars
        """
        bars_data: Dict[str, Dict[Timeframe, Bar]] = {}
        for symbol, data_point in datapoints.items():
            bars_data[symbol] = {}
            # Update all timeframes
            for timeframe in data_point:
                ohlcv_data = data_point[timeframe]

                # If the series contains any NaN values, skip this iteration
                if ohlcv_data.isna().any():
                    return {}

                bar = self._create_bar(symbol, timeframe, ohlcv_data)

                bars_data[symbol][timeframe] = bar

        return bars_data

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dict[str, Any]: A dictionary containing information about the loaded data,
            including symbols, timeframes, date range, and total number of bars.
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
            symbol (str): The symbol for which to retrieve data.
            timeframe (Timeframe): The timeframe of the data.
            n_bars (int, optional): The number of recent bars to retrieve. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the requested market data.
        """
        return self._dataview.get_data(symbol, timeframe, n_bars)

    # endregion

    # region Strategy Management

    def add_strategy(
        self,
        strategy: Type[Strategy],
        *args,
        **kwargs,
    ) -> None:
        """
        Add a trading strategy to the engine.

        This method instantiates a strategy class and initializes it with the provided parameters.

        Args:
            strategy (Type[Strategy]): The strategy class to be instantiated.
            *args: Positional arguments for strategy initialization.
            **kwargs: Keyword arguments for strategy initialization.

        Raises:
            ValueError: If invalid arguments are provided or required parameters are missing.
        """
        # Process timeframes
        valid_timeframes = self._process_strategy_timeframes(args)

        # Prepare strategy initialization parameters
        init_parameters = self._prepare_strategy_parameters(strategy, kwargs)

        # Initialize the strategy
        try:
            strategy_instance = strategy(**init_parameters)
        except Exception as e:
            logger_main.log_and_raise(f"Failed to initialize strategy: {str(e)}")

        strategy_instance.set_engine(self)

        # Store strategy and its timeframes
        self._strategies[strategy_instance._id] = strategy_instance
        self._strategy_timeframes[strategy_instance._id] = valid_timeframes

        logger_main.info(
            f"Added strategy: {strategy_instance.__class__.__name__} with ID: {strategy_instance._id}"
        )

    def _process_strategy_timeframes(self, args: Tuple[Any, ...]) -> List[Timeframe]:
        """
        Process and validate timeframes for a strategy.

        Args:
            args (Tuple[Any, ...]): Positional arguments that may contain timeframes.

        Returns:
            List[Timeframe]: A list of valid timeframes for the strategy.
        """
        valid_timeframes = []
        for arg in args:
            if isinstance(arg, Timeframe):
                valid_timeframes.append(arg)
            elif isinstance(arg, str):
                try:
                    timeframe = Timeframe(arg)
                    valid_timeframes.append(timeframe)
                    logger_main.info(
                        f"Converted argument `{arg}` to a Timeframe object `{repr(timeframe)}`"
                    )
                except Exception as e:
                    logger_main.warning(
                        f"Failed to convert argument `{arg}` to a Timeframe object. Error: {e}"
                    )

        if not valid_timeframes:
            valid_timeframes.append(self.default_timeframe)
            logger_main.info(
                f"No valid timeframes provided. Using default timeframe: {self.default_timeframe}"
            )

        return valid_timeframes

    def _prepare_strategy_parameters(
        self, strategy: Type[Strategy], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare initialization parameters for a strategy.

        Args:
            strategy (Type[Strategy]): The strategy class to prepare parameters for.
            kwargs (Dict[str, Any]): Keyword arguments provided for strategy initialization.

        Returns:
            Dict[str, Any]: A dictionary of initialization parameters for the strategy.

        Raises:
            ValueError: If a required parameter is missing.
        """
        init_parameters = {}
        strategy_signature = inspect.signature(strategy.__init__)
        for param_name, param in strategy_signature.parameters.items():
            if param_name == "self":
                continue
            if param_name in kwargs:
                init_parameters[param_name] = kwargs[param_name]
            elif param.default is not param.empty:
                init_parameters[param_name] = param.default
            else:
                logger_main.log_and_raise(
                    ValueError(f"Missing required parameter: {param_name}")
                )
        return init_parameters

    def remove_strategy(self, strategy_id: str) -> None:
        """
        Remove a trading strategy from the engine.

        Args:
            strategy_id (str): The ID of the strategy to be removed.
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            logger_main.info(f"Removed strategy with ID: {strategy_id}")
        else:
            logger_main.info(f"Strategy not found with ID: {strategy_id}")

    def get_strategy_by_id(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by its ID.

        Args:
            strategy_id (str): The ID of the strategy to retrieve.

        Returns:
            Optional[Strategy]: The requested strategy, or None if not found.
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
            logger_main.info(f"Updated parameters for strategy {strategy_id}")
        else:
            logger_main.info(f"Strategy with ID {strategy_id} not found.")

    def log_strategy_activity(self, strategy_id: str, message: str) -> None:
        """
        Log activity for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy logging the activity.
            message (str): The message to be logged.
        """
        logger_main.info(f"Strategy {strategy_id}: {message}")

    # endregion

    # region Backtesting Execution

    def run(self) -> Reporter:
        """
        Execute the backtest and return the Reporter object for analysis.

        This method runs the entire backtesting process, including data preparation,
        strategy execution, and portfolio updates.

        Returns:
            Reporter: The Reporter object containing all performance metrics and analysis tools.

        Raises:
            ValueError: If data validation fails or if the configuration is not set.
        """
        # Clear log file
        clear_log_files()

        if self._is_running:
            logger_main.info("Backtest is already running.")
            return self.reporter if self.reporter else Reporter(self.portfolio)

        if not self._config:
            logger_main.log_and_raise(
                ValueError(
                    "Configuration not set. Call set_config before running the backtest."
                )
            )

        try:
            self._dataview.align_all_data()

            if not self.validate_data():
                logger_main.log_and_raise(
                    ValueError("Data validation failed. Cannot start backtest.")
                )

            self._is_running = True
            self._initialize_backtest()
            logger_main.info("Starting backtest...")

            for timestamp, data_point in self._dataview:
                # Skip if data point contains missing data
                if not data_point:
                    continue

                self._current_timestamp = timestamp
                self._current_market_data = self._create_bars_data(data_point)
                self._process_timestamp(timestamp, self._current_market_data)

                if self._check_termination_condition():
                    break

            # self._finalize_backtest()

            # Initialize the Reporter with the final portfolio state
            self.reporter = Reporter(self.portfolio)
            logger_main.info("Backtest completed. Reporter initialized.")

        except Exception as e:
            logger_main.log_and_raise(f"Error during backtest: {str(e)}")
            raise
        finally:
            self._is_running = False

        return self.reporter

    def _initialize_backtest(self) -> None:
        """
        Initialize all components for the backtest.

        This method sets up the Portfolio and each strategy with the appropriate data and timeframes.
        It also ensures that the Reporter is reset to None at the start of each backtest.
        """

        margin_ratio = ExtendedDecimal(str(self._config.get("margin_ratio", "0.01")))

        # Create risk_manager_config dynamically
        risk_manager_config = {
            "initial_capital": ExtendedDecimal(
                str(self._config.get("initial_capital", 100000))
            ),
            "max_risk": ExtendedDecimal(
                str(self._config.get("max_position_size", "1"))
            ),
            "max_risk_per_trade": ExtendedDecimal(
                str(self._config.get("max_risk_per_trade", "1"))
            ),
            "max_risk_per_symbol": ExtendedDecimal(
                str(self._config.get("max_risk_per_symbol", "1"))
            ),
            "max_drawdown": ExtendedDecimal(
                str(self._config.get("max_drawdown", "0.9"))
            ),
            "var_confidence_level": float(
                self._config.get("var_confidence_level", 0.95)
            ),
            "margin_ratio": margin_ratio,
            "margin_call_threshold": ExtendedDecimal(
                str(self._config.get("margin_call_threshold", "0.01"))
            ),
        }

        # Initialize the Portfolio object
        self.portfolio = Portfolio(
            self,
            initial_capital=risk_manager_config["initial_capital"],
            commission_rate=ExtendedDecimal(
                str(self._config.get("commission_rate", "0.000"))
            ),
            margin_ratio=margin_ratio,
            risk_manager_config=risk_manager_config,
        )

        # Reset the Reporter
        self.reporter = None

        default_timeframe = self._dataview.lowest_timeframe
        for strategy_id, strategy in self._strategies.items():
            # Get the strategy timeframes
            strategy_timeframes = self._strategy_timeframes.get(
                strategy_id, [default_timeframe]
            )
            strategy.initialize(self._dataview.symbols, strategy_timeframes)
        logger_main.info("Backtest initialized.")

    def _process_timestamp(
        self, timestamp: pd.Timestamp, data_point: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Process a single timestamp in the backtest.

        This method handles all operations for a given timestamp, including
        updating the portfolio, processing strategy data, and notifying strategies.

        Args:
            timestamp (pd.Timestamp): The current timestamp being processed.
            data_point (Dict[str, Dict[Timeframe, Bar]]): The market data for this timestamp.
        """
        # Update portfolio with current market data
        self.portfolio.update(timestamp, data_point)

        # Update each strategy's data and run strategy logic
        for strategy in self._strategies.values():
            self._update_strategy_data(strategy, data_point)
            strategy._on_data()

        # Notify strategies of updates
        self._notify_strategies()

        # # Process any remaining orders in the order execution manager
        # self.portfolio.order_manager.process_orders(timestamp, data_point)

        # Clear updated orders and trades in the portfolio
        self.portfolio.order_manager.clear_updated_orders()
        self.portfolio.trade_manager.clear_updated_trades()

    def _update_strategy_data(
        self, strategy: Strategy, data_point: Dict[str, Dict[Timeframe, Bar]]
    ) -> None:
        """
        Update a strategy's data with the current market data.

        Args:
            strategy (Strategy): The strategy to update.
            data_point (Dict[str, Dict[Timeframe, Bar]]): The current market data.
        """
        for symbol in strategy.datas:
            for timeframe in strategy._strategy_timeframes:
                if self._dataview.is_original_data_point(
                    symbol, timeframe, self._current_timestamp
                ):
                    bar = data_point[symbol][timeframe]
                    strategy.datas[symbol].add_bar(bar)

    def _check_termination_condition(self) -> bool:
        """
        Check if the backtest should be terminated based on certain conditions.

        Returns:
            bool: True if the backtest should be terminated, False otherwise.
        """
        if self.portfolio.get_account_value() < self._config.get("min_equity", 0):
            logger_main.warning(
                "Portfolio value dropped below minimum equity. Terminating backtest."
            )
            return True
        return False

    def _finalize_backtest(self) -> None:
        """
        Perform final operations after the backtest is complete.

        This method closes all remaining positions and performs any necessary cleanup.
        It uses the current timestamp to ensure accurate position closing.
        """
        if self.portfolio is not None and self._current_timestamp is not None:
            self.portfolio.close_all_positions(self._current_timestamp)
            logger_main.info("Finalized backtest. Closed all remaining positions.")
        else:
            logger_main.warning(
                "Unable to finalize backtest. Portfolio or current timestamp is not set."
            )

    # endregion

    # region Order and Trade Management
    def _notify_order_update(self, order: Order) -> None:
        """
        Notify the relevant strategy of an order updated.

        Args:
            order (Order): The order that was updated.
        """
        strategy = self.get_strategy_by_id(order.details.strategy_id)
        if strategy:
            strategy.on_order_update(order)

    def _notify_strategies(self) -> None:
        """
        Notify strategies of updates to orders and trades.

        This method informs all strategies about any changes to their orders or trades
        that occurred during the current timestamp processing.
        """
        for order in self.portfolio.order_manager.get_updated_orders():
            strategy = self.get_strategy_by_id(order.details.strategy_id)
            if strategy:
                strategy.on_order_update(order)

        for trade in self.portfolio.trade_manager.updated_trades:
            strategy = self.get_strategy_by_id(trade.strategy_id)
            if strategy:
                strategy.on_trade_update(trade)

    def _notify_trade_update(self, trade: Trade) -> None:
        """
        Notify the relevant strategy of a trade update.

        Args:
            trade (Trade): The updated trade.
        """
        strategy = self.get_strategy_by_id(trade.strategy_id)
        if strategy:
            strategy.on_trade_update(trade)

    # endregion

    # region Portfolio and Account Information
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing the current timestamp, portfolio state,
            running status, and list of strategies.
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
            Dict[str, Any]: A dictionary containing various performance metrics for the strategy.
        """
        strategy = self.get_strategy_by_id(strategy_id)
        if not strategy:
            logger_main.info(f"Strategy with ID {strategy_id} not found.")
            return {}

        trades = self.get_trades_for_strategy(strategy_id)
        return {
            "total_trades": len(trades),
            "winning_trades": sum(1 for trade in trades if trade.metrics.pnl > 0),
            "losing_trades": sum(1 for trade in trades if trade.metrics.pnl < 0),
            "total_pnl": sum(trade.metrics.pnl for trade in trades),
            "realized_pnl": sum(trade.metrics.realized_pnl for trade in trades),
            "unrealized_pnl": sum(trade.metrics.unrealized_pnl for trade in trades),
            # Add more metrics as needed
        }

    # endregion

    def get_current_data(self, symbol: str) -> Bar:
        """
        Get the current data for a given symbol at the lowest available timeframe.

        This method retrieves the most recent bar from the current market data
        for the specified symbol at the lowest timeframe.

        Args:
            symbol (str): The symbol to get the current data for.

        Returns:
            Bar: The most recent bar for the symbol at the lowest timeframe.

        Raises:
            ValueError: If the symbol is not found in the current market data or if there's no data for the symbol.
        """
        if self._current_market_data is None:
            logger_main.log_and_raise(ValueError("No current market data available."))

        if symbol not in self._current_market_data:
            logger_main.log_and_raise(
                ValueError(f"Symbol {symbol} not found in the current market data")
            )

        # Get the lowest timeframe data for the symbol
        lowest_timeframe = min(self._current_market_data[symbol].keys())
        current_bar = self._current_market_data[symbol][lowest_timeframe]

        if current_bar is None:
            logger_main.log_and_raise(
                ValueError(f"No data available for symbol {symbol}")
            )

        return current_bar
