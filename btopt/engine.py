import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from .data.bar import Bar
from .data.dataloader import BaseDataLoader
from .data.dataview import DataView
from .data.timeframe import Timeframe
from .log_config import clear_log_file, logger_main
from .order import Order
from .portfolio import Portfolio
from .reporter import Reporter
from .strategy.strategy import Strategy
from .trade import Trade
from .util.decimal import ExtendedDecimal


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
        self.portfolio: Portfolio = Portfolio(engine=self)
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
            raise ValueError("No data has been added to the DataView yet.")
        return self._dataview.lowest_timeframe

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the backtest.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self._config = config
        self.portfolio.set_config(config)
        logger_main.info("Backtest configuration set.")

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
            logger_main.info("No data has been loaded.", level="warning")
            return False

        if len(self._dataview.symbols) == 0:
            logger_main.info("No symbols found in the loaded data.", level="warning")
            return False

        if len(self._dataview.timeframes) == 0:
            logger_main.info("No timeframes found in the loaded data.", level="warning")
            return False

        date_range = self._dataview.get_data_range()
        if date_range[0] == date_range[1]:
            logger_main.info(
                "Insufficient data: only one data point available.", level="warning"
            )
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

        Args:
            strategy (Type[Strategy]): The strategy class to be instantiated.
            *args: Positional arguments for strategy initialization.
            **kwargs: Keyword arguments for strategy initialization.

        Raises:
            ValueError: If invalid arguments are provided or required parameters are missing.
        """
        # Process timeframes
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
                        f"Failed to convert argument `{arg}` to a Timeframe object. Error: {e}",
                    )

        # Use default timeframe if none provided
        if not valid_timeframes:
            valid_timeframes.append(self.default_timeframe)
            logger_main.info(
                f"No valid timeframes provided. Using default timeframe: {self.default_timeframe}",
            )

        # Prepare strategy initialization parameters
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

        # Initialize the strategy
        try:
            strategy_instance = strategy(**init_parameters)
        except Exception as e:
            logger_main.log_and_raise(
                ValueError(f"Failed to initialize strategy: {str(e)}")
            )
        strategy_instance.set_engine(self)

        # Store strategy and its timeframes
        self._strategies[strategy_instance._id] = strategy_instance
        self._strategy_timeframes[strategy_instance._id] = valid_timeframes

        logger_main.info(
            f"Added strategy: {strategy_instance.__class__.__name__} with ID: {strategy_instance._id}"
        )

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
            logger_main.info(
                f"Strategy not found with ID: {strategy_id}", level="warning"
            )

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
            logger_main.info(
                f"Strategy with ID {strategy_id} not found.", level="warning"
            )

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
        strategy execution, and portfolio updates. After the backtest is complete,
        it initializes the Reporter with the final portfolio state and returns it
        for further analysis.

        Returns:
            ReporterType: The Reporter object containing all performance metrics and analysis tools.

        Raises:
            ValueError: If data validation fails.
            Exception: If an error occurs during the backtest.
        """
        # Clear log file
        clear_log_file()

        if self._is_running:
            logger_main.info("Backtest is already running.", level="warning")
            return self.reporter if self.reporter else Reporter(self.portfolio, self)

        try:
            self._dataview.align_all_data()

            if not self.validate_data():
                logger_main.log_and_raise(
                    ValueError("Data validation failed. Cannot start backtest.")
                )

            self._is_running = True
            self._initialize_backtest()
            logger_main.info("Starting backtest...")

            # Ensure all orders have a valid timeframe before starting the backtest
            self._validate_order_timeframes()

            for timestamp, data_point in self._dataview:
                # Data point contains missing data
                if not data_point:
                    continue

                self._current_timestamp = timestamp
                self._current_market_data = self._create_bars_data(data_point)
                self._process_timestamp(timestamp, self._current_market_data)

                if self._check_termination_condition():
                    break

            # Initialize the Reporter with the final portfolio state
            self.reporter = Reporter(self.portfolio, self)
            logger_main.info("Backtest completed. Reporter initialized.")

        except Exception as e:
            logger_main.log_and_raise(Exception(f"Error during backtest: {str(e)}"))
        finally:
            self._is_running = False

        return self.reporter

    def _initialize_backtest(self) -> None:
        """
        Initialize all components for the backtest.

        This method sets up each strategy with the appropriate data and timeframes.
        It also ensures that the Reporter is reset to None at the start of each backtest.
        """
        # Reset the Reporter
        self.reporter = None

        default_timeframe = self._dataview.lowest_timeframe
        for strategy_id, strategy in self._strategies.items():
            # Get the strategy timeframes
            strategy_timeframes = self._strategy_timeframes.get(
                strategy_id, [default_timeframe]
            )
            strategy.initialize(
                self._dataview.symbols,
                strategy_timeframes,
            )
        logger_main.info("Backtest initialized.")

    def _process_timestamp(
        self,
        timestamp: pd.Timestamp,
        data_point: Dict[str, Dict[Timeframe, Bar]],
    ) -> None:
        # Process order fills
        self._process_order_fills(data_point)

        # Update each strategy datas
        for strategy in self._strategies.values():
            # Load new data for all symbols
            for symbol in strategy.datas:
                # Update all timeframes
                for timeframe in strategy._strategy_timeframes:
                    if self._dataview.is_original_data_point(
                        symbol, timeframe, timestamp
                    ):
                        bar = data_point[symbol][timeframe]
                        strategy.datas[bar.ticker].add_bar(bar)

            # Run strategy.on_data
            strategy._on_data()

        # Update portfolio
        self.portfolio.update(timestamp, data_point)

        # Notify strategies of updates
        self._notify_strategies()

    def _process_order_fills(self, data_point: Dict[str, Dict[Timeframe, Bar]]) -> None:
        for order in self.portfolio.pending_orders + self.portfolio.limit_exit_orders:
            symbol = order.details.ticker
            current_bar = data_point[symbol][order.details.timeframe].close

            is_filled, fill_price = order.is_filled(current_bar)
            if is_filled:
                executed, trade = self.portfolio.execute_order(order, fill_price)
                if executed:
                    self._notify_order_fill(order, trade)

    def _check_termination_condition(self) -> bool:
        """
        Check if the backtest should be terminated based on certain conditions.

        Returns:
            bool: True if the backtest should be terminated, False otherwise.
        """
        if self.portfolio.calculate_equity() < self._config.get("min_equity", 0):
            logger_main.warning(
                "Portfolio value dropped below minimum equity. Terminating backtest.",
            )
            return True
        return False

    def _finalize_backtest(self) -> None:
        """
        Perform final operations after the backtest is complete.

        This method closes all remaining trades and performs any necessary cleanup.
        """
        self.close_all_positions()
        logger_main.info("Finalized backtest. Closed all remaining trades.")

    def _validate_order_timeframes(self) -> None:
        """
        Ensure all pending orders have a valid timeframe set.

        This method checks all pending orders and sets the default timeframe
        if the order's timeframe is None. It logs a warning for each order
        that requires a timeframe update.
        """
        for order in self.portfolio.pending_orders + self.portfolio.limit_exit_orders:
            if order.details.timeframe is None:
                order.details.timeframe = self.default_timeframe
                logger_main.warning(
                    f"Updated order {order.id} for {order.details.ticker} to use default timeframe {self.default_timeframe}",
                )

    # endregion

    # region Order and Trade Management

    def create_order(
        self,
        strategy_id: str,
        symbol: str,
        direction: Order.Direction,
        size: float,
        order_type: Order.ExecType,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Order, List[Order]]:
        """
        Create an order and associated child orders.

        Args:
            strategy_id (str): The ID of the strategy creating the order.
            symbol (str): The symbol to trade.
            direction (Order.Direction): The direction of the trade (LONG or SHORT).
            size (float): The size of the order.
            order_type (Order.ExecType): The type of order (e.g., MARKET, LIMIT).
            price (Optional[float], optional): The price for limit orders. Defaults to None.
            stop_loss (Optional[float], optional): The stop-loss price. Defaults to None.
            take_profit (Optional[float], optional): The take-profit price. Defaults to None.
            **kwargs: Additional keyword arguments for the order.

        Returns:
            Tuple[Order, List[Order]]: The parent order and a list of child orders (stop-loss and take-profit).
        """
        # Use the default timeframe if not provided in kwargs
        if ("timeframe" not in kwargs) or kwargs["timeframe"] is None:
            kwargs["timeframe"] = self.default_timeframe
            logger_main.info(
                f"Using default timeframe {self.default_timeframe} for order on {symbol}",
            )

        parent_order = self.portfolio.create_order(
            symbol, direction, size, order_type, price, **kwargs
        )
        child_orders = self.create_limit_exit_orders(
            parent_order, stop_loss, take_profit
        )
        return parent_order, child_orders

    def create_limit_exit_orders(
        self,
        parent_order: Order,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> List[Order]:
        """
        Create child orders for stop-loss and take-profit.

        Args:
            parent_order (Order): The parent order.
            stop_loss (Optional[float]): The stop-loss price.
            take_profit (Optional[float]): The take-profit price.

        Returns:
            List[Order]: A list of child orders (stop-loss and take-profit).
        """
        child_orders = []
        if stop_loss:
            stop_loss_order = self.portfolio.create_order(
                parent_order.details.ticker,
                Order.Direction.SHORT
                if parent_order.details.direction == Order.Direction.LONG
                else Order.Direction.LONG,
                parent_order.details.size,
                Order.ExecType.STOP,
                stop_loss,
                parent_id=parent_order.id,
            )
            child_orders.append(stop_loss_order)

        if take_profit:
            take_profit_order = self.portfolio.create_order(
                parent_order.details.ticker,
                Order.Direction.SHORT
                if parent_order.details.direction == Order.Direction.LONG
                else Order.Direction.LONG,
                parent_order.details.size,
                Order.ExecType.LIMIT,
                take_profit,
                parent_id=parent_order.id,
            )
            child_orders.append(take_profit_order)

        return child_orders

    def cancel_order(self, strategy_id: str, order: Order) -> bool:
        """
        Cancel an existing order.

        Args:
            strategy_id (str): The ID of the strategy cancelling the order.
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.
        """
        if order.details.strategy_id != strategy_id:
            logger_main.warning(
                f"Strategy {strategy_id} attempted to cancel an order it didn't create.",
            )
            return False

        success = self.portfolio.cancel_order(order)
        return success

    def close_positions(self, strategy_id: str, symbol: Optional[str] = None) -> bool:
        """
        Close all positions for a strategy, or for a specific symbol if provided.

        Args:
            strategy_id (str): The ID of the strategy closing the positions.
            symbol (Optional[str], optional): The specific symbol to close positions for. Defaults to None.

        Returns:
            bool: True if positions were successfully closed, False otherwise.
        """
        success = self.portfolio.close_positions(strategy_id=strategy_id, symbol=symbol)
        return success

    def close_all_positions(self) -> None:
        """
        Close all open positions in the portfolio.
        """
        if self._current_timestamp is None:
            logger_main.log_and_raise(ValueError("Backtest has not been run yet."))

        self.portfolio.close_all_positions(self._current_timestamp)
        logger_main.info("Closed all open positions.")

    def _notify_order_fill(self, order: Order, trade: Optional[Trade]) -> None:
        """
        Notify the relevant strategy of an order fill.

        Args:
            order (Order): The order that was filled.
            trade (Optional[Trade]): The resulting trade, if any.
        """
        strategy = self.get_strategy_by_id(order.details.strategy_id)
        if strategy:
            strategy.on_order_fill(order, trade)

    def _notify_strategies(self) -> None:
        """
        Notify strategies of updates to orders and trades.

        This method is called after processing each timestamp to inform strategies
        of any changes to their orders or trades.
        """
        for order in self.portfolio.updated_orders:
            self._notify_order_fill(order, None)
        for trade in self.portfolio.updated_trades:
            self._notify_trade_update(trade)

    def _notify_trade_update(self, trade: Trade) -> None:
        """
        Notify the relevant strategy of a trade update.

        Args:
            trade (Trade): The updated trade.
        """
        strategy = self.get_strategy_by_id(trade.strategy_id)
        if strategy:
            strategy._on_trade_update(trade)

    # endregion

    # region Portfolio and Account Information

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            List[Trade]: A list of trades associated with the strategy.
        """
        return self.portfolio.get_trades_for_strategy(strategy_id)

    def get_account_value(self) -> ExtendedDecimal:
        """
        Get the current total account value (equity).

        Returns:
            ExtendedDecimal: The current account value.
        """
        return self.portfolio.get_account_value()

    def get_position_size(self, symbol: str) -> ExtendedDecimal:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            ExtendedDecimal: The current position size (positive for long, negative for short).
        """
        return self.portfolio.get_position_size(symbol)

    def get_available_margin(self) -> ExtendedDecimal:
        """
        Get the available margin for new trades.

        Returns:
            ExtendedDecimal: The amount of available margin.
        """
        return self.portfolio.get_available_margin()

    def get_open_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        """
        Get open trades, optionally filtered by strategy ID.

        Args:
            strategy_id (Optional[str], optional): The ID of the strategy to filter trades for. Defaults to None.

        Returns:
            List[Trade]: A list of open trades.
        """
        return self.portfolio.get_open_trades(strategy_id)

    def get_closed_trades(self, strategy_id: Optional[str] = None) -> List[Trade]:
        """
        Get closed trades, optionally filtered by strategy ID.

        Args:
            strategy_id (Optional[str], optional): The ID of the strategy to filter trades for. Defaults to None.

        Returns:
            List[Trade]: A list of closed trades.
        """
        return self.portfolio.get_closed_trades(strategy_id)

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
            logger_main.info(
                f"Strategy with ID {strategy_id} not found.", level="warning"
            )
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
