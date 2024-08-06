from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .data.bar import Bar
from .data.dataloader import BaseDataLoader
from .data.dataview import DataView
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order
from .portfolio import Portfolio
from .strategy.strategy import Strategy
from .trade import Trade


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
        self.portfolio: Portfolio = Portfolio()
        self._strategies: Dict[str, Strategy] = {}
        self._current_timestamp: Optional[pd.Timestamp] = None
        self._is_running: bool = False
        self._config: Dict[str, Any] = {}
        self._strategy_timeframes: Dict[str, Timeframe] = {}

    # region Initialization and Configuration

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the backtest.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
        """
        self._config = config
        self.portfolio.set_config(config)
        logger_main.log_and_print("Backtest configuration set.", level="info")

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
        logger_main.log_and_print("Engine reset to initial state.", level="info")

    def validate_data(self) -> bool:
        """
        Validate the loaded data to ensure it's sufficient for backtesting.

        Returns:
            bool: True if the data is valid and sufficient, False otherwise.
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
        logger_main.log_and_print(
            f"Added data for {len(dataloader.dataframes)} symbols.", level="info"
        )

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

        for ticker, data in dataloader.dataframes.items():
            self._dataview.resample_data(
                symbol=ticker,
                from_timeframe=dataloader.timeframe,
                to_timeframe=timeframe,
                df=data,
            )

        self.add_data(dataloader)
        logger_main.log_and_print(
            f"Resampled data for {len(dataloader.dataframes)} symbols to {timeframe}.",
            level="info",
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
            open=Decimal(str(data["open"])),
            high=Decimal(str(data["high"])),
            low=Decimal(str(data["low"])),
            close=Decimal(str(data["close"])),
            volume=int(data["volume"]),
            timestamp=self._current_timestamp,
            timeframe=timeframe,
            ticker=symbol,
        )

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
        self, strategy: Strategy, primary_timeframe: Optional[Timeframe] = None
    ) -> None:
        """
        Add a trading strategy to the engine.

        Args:
            strategy (Strategy): The strategy to be added.
            primary_timeframe (Optional[Timeframe], optional): The primary timeframe for the strategy.
                If not provided, the lowest available timeframe will be used.
        """
        strategy.set_engine(self)
        self._strategies[strategy._id] = strategy
        if primary_timeframe is not None:
            self._strategy_timeframes[strategy._id] = primary_timeframe
        logger_main.log_and_print(
            f"Added strategy: {strategy.__class__.__name__}", level="info"
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
            strategy_id (str): The ID of the strategy logging the activity.
            message (str): The message to be logged.
        """
        logger_main.log_and_print(f"Strategy {strategy_id}: {message}", level="info")

    # endregion

    # region Backtesting Execution

    def run(self) -> Dict[str, Any]:
        """
        Execute the backtest and return the results.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results, including performance metrics,
            trade history, and equity curve.

        Raises:
            ValueError: If data validation fails.
            Exception: If an error occurs during the backtest.
        """
        if self._is_running:
            logger_main.log_and_print("Backtest is already running.", level="warning")
            return {}
        try:
            self._dataview.align_all_data()

            if not self.validate_data():
                logger_main.log_and_raise(
                    ValueError("Data validation failed. Cannot start backtest.")
                )

            self._is_running = True
            self._initialize_backtest()
            logger_main.log_and_print("Starting backtest...", level="info")

            for timestamp, data_point in self._dataview:
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
        """
        Initialize all components for the backtest.

        This method sets up each strategy with the appropriate data and timeframes.
        """
        default_timeframe = min(self._dataview.timeframes)
        for strategy_id, strategy in self._strategies.items():
            primary_timeframe = self._strategy_timeframes.get(
                strategy_id, default_timeframe
            )
            strategy.initialize(
                self._dataview.symbols,
                self._dataview.timeframes,
                primary_timeframe,
            )
        logger_main.log_and_print("Backtest initialized.", level="info")

    def _process_timestamp(
        self,
        timestamp: pd.Timestamp,
        data_point: Dict[str, Dict[Timeframe, pd.Series]],
    ) -> None:
        """
        Process a single timestamp across all symbols and timeframes.

        This method handles order fills, updates strategies, and manages the portfolio
        for each timestamp in the backtest.

        Args:
            timestamp (pd.Timestamp): The current timestamp being processed.
            data_point (Dict[str, Dict[Timeframe, pd.Series]]): Market data for the current timestamp.
        """
        # Process order fills
        self._process_order_fills(data_point)

        # Update strategies
        for strategy_id, strategy in self._strategies.items():
            for symbol in self._dataview.symbols:
                if self._dataview.is_original_data_point(
                    symbol, strategy.primary_timeframe, timestamp
                ):
                    ohlcv_data = data_point[symbol][strategy.primary_timeframe]
                    bar = self._create_bar(
                        symbol, strategy.primary_timeframe, ohlcv_data
                    )
                    strategy.process_bar(bar)

        # Update portfolio
        self.portfolio.update(timestamp, data_point)

        # Notify strategies of updates
        self._notify_strategies()

    def _process_order_fills(
        self, data_point: Dict[str, Dict[Timeframe, pd.Series]]
    ) -> None:
        """
        Process order fills based on current market data.

        This method checks all pending and limit exit orders against the current market data
        to determine if they should be filled.

        Args:
            data_point (Dict[str, Dict[Timeframe, pd.Series]]): Current market data for all symbols and timeframes.
        """
        for order in self.portfolio.pending_orders + self.portfolio.limit_exit_orders:
            symbol = order.details.ticker
            current_price = data_point[symbol][order.details.timeframe]["close"]
            is_filled, fill_price = order.is_filled(current_price)
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
            logger_main.log_and_print(
                "Portfolio value dropped below minimum equity. Terminating backtest.",
                level="warning",
            )
            return True
        return False

    def _finalize_backtest(self) -> None:
        """
        Perform final operations after the backtest is complete.

        This method closes all remaining trades and performs any necessary cleanup.
        """
        self.portfolio.close_all_trades(self._current_timestamp, self._dataview)
        logger_main.log_and_print(
            "Finalized backtest. Closed all remaining trades.", level="info"
        )

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate and return the final backtest results.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics, trade history, and equity curve.
        """
        return {
            "performance_metrics": self.portfolio.get_performance_metrics(),
            "trade_history": self.portfolio.get_trade_history(),
            "equity_curve": self.portfolio.get_equity_curve(),
        }

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
            strategy_id (str): The ID of the strategy closing the positions.
            symbol (Optional[str], optional): The specific symbol to close positions for. Defaults to None.

        Returns:
            bool: True if positions were successfully closed, False otherwise.
        """
        success = self.portfolio.close_positions(strategy_id=strategy_id, symbol=symbol)
        return success

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
        for order in self.portfolio.get_updated_orders():
            self._notify_order_fill(order, None)
        for trade in self.portfolio.get_updated_trades():
            self._notify_trade_update(trade)

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

    def get_trades_for_strategy(self, strategy_id: str) -> List[Trade]:
        """
        Get all trades for a specific strategy.

        Args:
            strategy_id (str): The ID of the strategy.

        Returns:
            List[Trade]: A list of trades associated with the strategy.
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
            Decimal: The current position size (positive for long, negative for short).
        """
        return self.portfolio.get_position_size(symbol)

    def get_available_margin(self) -> Decimal:
        """
        Get the available margin for new trades.

        Returns:
            Decimal: The amount of available margin.
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
            logger_main.log_and_print(
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

    # region Utility Methods

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall performance metrics for the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.
        """
        return self.portfolio.get_performance_metrics()

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the equity curve data.
        """
        return self.portfolio.get_equity_curve()

    def get_drawdown_curve(self) -> pd.DataFrame:
        """
        Get the drawdown curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the drawdown curve data.
        """
        equity_curve = self.get_equity_curve()
        equity_curve["Drawdown"] = (
            equity_curve["equity"].cummax() - equity_curve["equity"]
        ) / equity_curve["equity"].cummax()
        return equity_curve[["timestamp", "Drawdown"]]

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete trade history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a trade.
        """
        return self.portfolio.get_trade_history()

    def save_results(self, filename: str) -> None:
        """
        Save backtest results to a file.

        Args:
            filename (str): The name of the file to save the results to.
        """
        results = self._generate_results()
        # Implementation depends on the desired file format (e.g., JSON, CSV, pickle)
        # For example, using JSON:
        import json

        with open(filename, "w") as f:
            json.dump(results, f, default=str)
        logger_main.log_and_print(f"Backtest results saved to {filename}", level="info")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load backtest results from a file.

        Args:
            filename (str): The name of the file to load the results from.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded backtest results.
        """
        # Implementation depends on the file format used in save_results
        # For example, using JSON:
        import json

        with open(filename, "r") as f:
            results = json.load(f)
        logger_main.log_and_print(
            f"Backtest results loaded from {filename}", level="info"
        )
        return results

    # endregion
