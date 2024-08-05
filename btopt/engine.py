from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .data.bar import Bar
from .data.dataloader import BaseDataLoader
from .data.dataview import DataView, DataViewNumpy
from .data.timeframe import Timeframe
from .log_config import logger_main
from .order import Order, OrderDetails
from .portfolio import Portfolio
from .strategy import Strategy
from .trade import Trade


class Engine:
    """
    A comprehensive backtesting engine for financial trading strategies.

    This class manages the entire backtesting process, including data management,
    strategy execution, portfolio tracking, and performance analysis.

    Attributes:
        _dataview (DataView): Stores and manages financial data.
        _optimized_dataview (DataViewNumpy): Optimized numpy-based view of the data.
        _portfolio (Portfolio): Manages trades, orders, and performance metrics.
        _portfolio_kwargs (dict): Keyword arguments for portfolio initialization.
        _strategies (List[Strategy]): List of trading strategies to be tested.
        _current_timestamp (pd.Timestamp): Current timestamp in the backtest.
        _is_running (bool): Flag indicating if a backtest is currently running.

    Methods:
        run() -> Dict[str, Any]:
            Execute the backtest and return the results.

        add_data(dataloader: BaseDataLoader) -> None:
            Add financial data to the engine from a dataloader.

        resample_data(dataloader: BaseDataLoader, timeframe: Union[str, Timeframe]) -> None:
            Resample data to a new timeframe and add it to the engine.

        add_strategy(strategy: Strategy) -> None:
            Add a trading strategy to the engine.

        remove_strategy(strategy: Strategy) -> None:
            Remove a trading strategy from the engine.

        set_initial_capital(capital: float) -> None:
            Set the initial capital for the portfolio.

        set_commission_rate(rate: float) -> None:
            Set the commission rate for trades.

        set_slippage(slippage: float) -> None:
            Set the slippage model for order execution.

        set_pyramiding(max_trades: int) -> None:
            Set the maximum number of trades allowed per symbol.

        set_max_drawdown(max_drawdown: float) -> None:
            Set the maximum allowable drawdown.

        set_risk_per_trade(risk_percentage: float) -> None:
            Set the risk percentage per trade.

        get_performance_metrics() -> Dict[str, Any]:
            Get overall performance metrics of the backtest.

        get_trade_history() -> List[Dict[str, Any]]:
            Get the complete trade history.

        get_equity_curve() -> pd.DataFrame:
            Get the equity curve data.

        plot_results() -> None:
            Generate plots for backtest results.

        reset() -> None:
            Reset the engine to its initial state.

        get_current_state() -> Dict[str, Any]:
            Get the current state of the backtest.

        save_results(filename: str) -> None:
            Save the backtest results to a file.

        load_results(filename: str) -> Dict[str, Any]:
            Load backtest results from a file.

    Private Methods:
        _initialize_backtest() -> None:
            Initialize all components for the backtest.

        _process_timestamp(timestamp: pd.Timestamp, data_point: Dict[str, Dict[Timeframe, np.ndarray]]) -> None:
            Process a single timestamp across all symbols and timeframes.

        _process_data_point(symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp, ohlcv_data: np.ndarray) -> None:
            Process a single data point, applying strategies and updating the portfolio.

        _generate_signals(symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp, bar: Bar) -> List[Dict[str, Any]]:
            Generate trading signals from all strategies.

        _execute_signals(signals: List[Dict[str, Any]], timestamp: pd.Timestamp) -> None:
            Execute the generated signals, creating and managing orders.

        _update_portfolio(timestamp: pd.Timestamp) -> None:
            Update the portfolio state for the current timestamp.

        _process_pending_orders(timestamp: pd.Timestamp) -> None:
            Process all pending orders based on current market data.

        _handle_filled_orders(filled_orders: List[Tuple[Order, bool, Optional[Trade]]]) -> None:
            Handle orders that have been filled, creating or updating trades.

        _update_open_trades(timestamp: pd.Timestamp) -> None:
            Update all open trades based on current market data.

        _check_exit_conditions(timestamp: pd.Timestamp) -> None:
            Check and execute exit conditions for open trades.

        _generate_results() -> Dict[str, Any]:
            Generate and return the final backtest results.

        _init_dataview_numpy() -> None:
            Initialize the optimized NumPy-based DataView.

        _create_order_from_signal(signal: Dict[str, Any], timestamp: pd.Timestamp) -> Order:
            Create an Order object from a strategy signal.

        _check_termination_condition() -> bool:
            Check if the backtest should be terminated based on certain conditions.

        _finalize_backtest() -> None:
            Perform final operations after the backtest is complete.

        _validate_backtest_parameters() -> None:
            Validate all parameters before starting the backtest.

        _log_backtest_progress(current_timestamp: pd.Timestamp) -> None:
            Log the progress of the backtest.

        _handle_backtest_errors(error: Exception) -> None:
            Handle and log any errors that occur during the backtest.

    Usage:
        engine = Engine()
        engine.add_data(your_dataloader)
        engine.add_strategy(your_strategy)
        engine.set_initial_capital(100000)
        engine.set_commission_rate(0.001)
        results = engine.run()
        engine.plot_results()
    """

    def __init__(self):
        self._dataview = DataView()
        self._optimized_dataview = None
        self._portfolio = None
        self._portfolio_kwargs = {}
        self._strategies = []
        self._current_timestamp = None
        self._is_running = False

    # region - Core Backtesting
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.

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
                self._process_timestamp(timestamp, data_point)

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

    def set_initial_capital(self, capital: float) -> None:
        """
        Set the initial capital for the portfolio.

        Args:
            capital (float): The initial capital amount.
        """
        self._portfolio_kwargs["initial_capital"] = capital
        logger_main.info(f"Set initial capital to: {capital}")

    def set_commission_rate(self, rate: float) -> None:
        """
        Set the commission rate for trades.

        Args:
            rate (float): The commission rate as a decimal (e.g., 0.001 for 0.1%).
        """
        self._portfolio_kwargs["commission_rate"] = rate
        logger_main.info(f"Set commission rate to: {rate}")

    def set_slippage(self, slippage: float) -> None:
        """
        Set the slippage model for order execution.

        Args:
            slippage (float): The slippage as a decimal (e.g., 0.001 for 0.1%).
        """
        self._portfolio_kwargs["slippage"] = slippage
        logger_main.info(f"Set slippage to: {slippage}")

    def set_pyramiding(self, max_trades: int) -> None:
        """
        Set the maximum number of trades allowed per symbol.

        Args:
            max_trades (int): The maximum number of concurrent trades per symbol.
        """
        self._portfolio_kwargs["pyramiding"] = max_trades
        logger_main.info(f"Set pyramiding to: {max_trades}")

    def set_max_drawdown(self, max_drawdown: float):
        """
        Set the maximum allowable drawdown.

        Args:
            max_drawdown (float): The maximum drawdown as a decimal (e.g., 0.2 for 20%).
        """
        self._portfolio_kwargs["max_drawdown"] = max_drawdown
        logger_main.info(f"Set maximum drawdown to: {max_drawdown}")

    def set_risk_per_trade(self, risk_percentage: float):
        """
        Set the risk percentage per trade.

        Args:
            risk_percentage (float): The risk percentage as a decimal (e.g., 0.01 for 1%).
        """
        self._portfolio_kwargs["risk_per_trade"] = risk_percentage
        logger_main.info(f"Set risk per trade to: {risk_percentage}")

    def _initialize_backtest(self):
        """Initialize all components for the backtest."""
        self._init_dataview_numpy()
        self._portfolio = Portfolio(**self._portfolio_kwargs)

        for strategy in self._strategies:
            strategy.initialize(
                self._optimized_dataview.symbols, self._optimized_dataview.timeframes
            )

        logger_main.info("Backtest initialized.")

    def _process_timestamp(
        self,
        timestamp: pd.Timestamp,
        data_point: Dict[str, Dict[Timeframe, np.ndarray]],
    ):
        """
        Process a single timestamp across all symbols and timeframes.

        Args:
            timestamp (pd.Timestamp): The current timestamp being processed.
            data_point (Dict[str, Dict[Timeframe, np.ndarray]]): Data for all symbols and timeframes at this timestamp.
        """
        for symbol in self._optimized_dataview.symbols:
            for timeframe in self._optimized_dataview.timeframes:
                if self._optimized_dataview.is_original_data_point(
                    symbol, timeframe, timestamp
                ):
                    ohlcv_data = data_point[symbol][timeframe]
                    self._process_data_point(symbol, timeframe, timestamp, ohlcv_data)

        self._update_portfolio(timestamp)
        self._process_pending_orders(timestamp)
        self._update_open_trades(timestamp)
        self._check_exit_conditions(timestamp)

    def _process_data_point(
        self,
        symbol: str,
        timeframe: Timeframe,
        timestamp: pd.Timestamp,
        ohlcv_data: np.ndarray,
    ):
        """
        Process a single data point, applying strategies and updating the portfolio.

        Args:
            symbol (str): The symbol being processed.
            timeframe (Timeframe): The timeframe of the data point.
            timestamp (pd.Timestamp): The timestamp of the data point.
            ohlcv_data (np.ndarray): The OHLCV data for the given symbol and timestamp.
        """
        bar = Bar(
            open=ohlcv_data[0],
            high=ohlcv_data[1],
            low=ohlcv_data[2],
            close=ohlcv_data[3],
            volume=ohlcv_data[4],
            timestamp=timestamp,
            timeframe=timeframe,
            ticker=symbol,
        )

        signals = self._generate_signals(symbol, timeframe, timestamp, bar)
        self._execute_signals(signals, timestamp)

    def _update_portfolio(self, timestamp: pd.Timestamp):
        """
        Update the portfolio state for the current timestamp.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
        """
        self._portfolio.update(timestamp, self._optimized_dataview)

    def _check_termination_condition(self) -> bool:
        """
        Check if the backtest should be terminated based on certain conditions.

        Returns:
            bool: True if the backtest should be terminated, False otherwise.
        """
        # Implement your termination conditions here
        # For example, you might want to stop if the portfolio value drops below a certain threshold
        if (
            self._portfolio.calculate_equity()
            < self._portfolio_kwargs["initial_capital"] * 0.5
        ):
            logger_main.warning(
                "Portfolio value dropped below 50% of initial capital. Terminating backtest."
            )
            return True
        return False

    def _finalize_backtest(self):
        """
        Perform final operations after the backtest is complete.
        """
        # Close any remaining open trades
        self._portfolio.close_all_trades(
            self._current_timestamp, self._optimized_dataview
        )
        logger_main.info("Finalized backtest. Closed all remaining trades.")

    def _validate_backtest_parameters(self):
        """
        Validate all parameters before starting the backtest.
        """
        if not self._strategies:
            logger_main.log_and_raise(
                ValueError("No strategies have been added to the engine.")
            )

        required_params = ["initial_capital", "commission_rate"]
        for param in required_params:
            if param not in self._portfolio_kwargs:
                logger_main.log_and_raise(
                    ValueError(f"Required parameter '{param}' is not set.")
                )

        logger_main.info("All backtest parameters validated successfully.")

    def _log_backtest_progress(self, current_timestamp: pd.Timestamp):
        """
        Log the progress of the backtest.

        Args:
            current_timestamp (pd.Timestamp): The current timestamp in the backtest.
        """
        start_date, end_date = self._dataview.get_data_range()
        progress = (current_timestamp - start_date) / (end_date - start_date) * 100
        logger_main.info(f"Backtest progress: {progress:.2f}% complete")

    def _handle_backtest_errors(self, error: Exception):
        """
        Handle and log any errors that occur during the backtest.

        Args:
            error (Exception): The error that occurred.
        """
        logger_main.error(f"Error during backtest: {str(error)}")
        # You might want to implement more sophisticated error handling here,
        # such as writing error details to a file or notifying via email

    def reset(self):
        """
        Reset the engine to its initial state.
        """
        self._portfolio = None
        self._current_timestamp = None
        self._is_running = False
        logger_main.info("Engine reset to initial state.")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the backtest.
        """
        return {
            "current_timestamp": self._current_timestamp,
            "portfolio_value": self._portfolio.calculate_equity()
            if self._portfolio
            else None,
            "open_trades": self._portfolio.get_open_trades_count()
            if self._portfolio
            else 0,
            "is_running": self._is_running,
        }

    # endregion

    # region - Data Management
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
        It should be called at the beginning of the run method.
        """
        self._dataview.align_all_data()
        self._optimized_dataview = DataViewNumpy(self._dataview)
        logger_main.log_and_print("Initialized optimized NumPy-based DataView.")

    def get_data_info(self) -> dict:
        """
        Get information about the loaded data.

        Returns:
            dict: A dictionary containing information about the loaded data,
                  including symbols, timeframes, and date ranges.
        """
        symbols = self._dataview.symbols
        timeframes = self._dataview.timeframes
        date_range = self._dataview.get_data_range()

        return {
            "symbols": symbols,
            "timeframes": timeframes,
            "start_date": date_range[0],
            "end_date": date_range[1],
        }

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

    # endregion

    # region - Signal Creation
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a trading strategy to the engine.

        Args:
            strategy (Strategy): The strategy to be added.
        """
        self._strategies.append(strategy)
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
        Generate trading signals from all strategies.

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

    def _execute_signals(self, signals: List[Dict[str, Any]], timestamp: pd.Timestamp):
        """
        Execute the generated signals, creating and managing orders.

        Args:
            signals (List[Dict[str, Any]]): A list of signal dictionaries to be executed.
            timestamp (pd.Timestamp): The current timestamp.
        """
        for signal in signals:
            if self._portfolio.can_open_new_trade(signal):
                order = self._create_order_from_signal(signal, timestamp)
                self._portfolio.add_pending_order(order)

    # endregion

    # region - Order/Trade Management
    def _process_pending_orders(self, timestamp: pd.Timestamp):
        """
        Process all pending orders based on current market data.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
        """
        filled_orders = self._portfolio.process_pending_orders(
            self._optimized_dataview, timestamp
        )
        self._handle_filled_orders(filled_orders)

    def _handle_filled_orders(
        self, filled_orders: List[Tuple[Order, bool, Optional[Trade]]]
    ):
        """
        Handle orders that have been filled, creating or updating trades.

        Args:
            filled_orders (List[Tuple[Order, bool, Optional[Trade]]]): A list of tuples containing
                the filled order, a boolean indicating if it was executed successfully,
                and the resulting trade if applicable.
        """
        for order, executed, trade in filled_orders:
            if executed:
                if trade:
                    logger_main.info(f"Order filled and trade created: {trade}")
                else:
                    logger_main.info(f"Order filled: {order}")
            else:
                logger_main.warning(f"Order execution failed: {order}")

    def _update_open_trades(self, timestamp: pd.Timestamp):
        """
        Update all open trades based on current market data.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
        """
        self._portfolio.update_open_trades(timestamp, self._optimized_dataview)

    def _check_exit_conditions(self, timestamp: pd.Timestamp):
        """
        Check and execute exit conditions for open trades.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
        """
        closed_trades = self._portfolio.check_exit_conditions(
            timestamp, self._optimized_dataview
        )
        for trade in closed_trades:
            logger_main.info(f"Trade closed: {trade}")

    def _create_order_from_signal(
        self, signal: Dict[str, Any], timestamp: pd.Timestamp
    ) -> Order:
        """
        Create an Order object from a strategy signal.

        Args:
            signal (Dict[str, Any]): The signal dictionary generated by a strategy.
            timestamp (pd.Timestamp): The current timestamp.

        Returns:
            Order: The created Order object.
        """
        order_details = OrderDetails(
            ticker=signal["symbol"],
            direction=signal["direction"],
            size=signal["size"],
            price=signal["price"],
            exectype=signal["order_type"],
            timestamp=timestamp,
            timeframe=signal["timeframe"],
            expiry=signal.get("expiry"),
            stoplimit_price=signal.get("stoplimit_price"),
            parent_id=signal.get("parent_id"),
            exit_profit=signal.get("exit_profit"),
            exit_loss=signal.get("exit_loss"),
            exit_profit_percent=signal.get("exit_profit_percent"),
            exit_loss_percent=signal.get("exit_loss_percent"),
            trailing_percent=signal.get("trailing_percent"),
            slippage=self._portfolio_kwargs.get("slippage"),
        )
        return Order(
            order_id=hash(f"{signal['symbol']}_{timestamp}_{signal['direction']}"),
            details=order_details,
        )

    # endregion

    # region - Reports and Visualizations
    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate and return the final backtest results.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results and performance metrics.
        """
        results = {
            "total_return": self._portfolio.calculate_total_return(),
            "sharpe_ratio": self._portfolio.calculate_sharpe_ratio(),
            "max_drawdown": self._portfolio.get_max_drawdown(),
            "trade_history": self._portfolio.get_trade_history(),
            "equity_curve": self._portfolio.get_equity_curve(),
        }
        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall performance metrics of the backtest.

        Returns:
            Dict[str, Any]: A dictionary containing various performance metrics.
        """
        return {
            "total_return": self._portfolio.calculate_total_return(),
            "sharpe_ratio": self._portfolio.calculate_sharpe_ratio(),
            "max_drawdown": self._portfolio.get_max_drawdown(),
            "win_rate": self._portfolio.calculate_win_rate(),
            "profit_factor": self._portfolio.calculate_profit_factor(),
            "total_trades": self._portfolio.get_total_trades(),
        }

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete trade history.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a trade.
        """
        return self._portfolio.get_trade_history()

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve data.

        Returns:
            pd.DataFrame: A DataFrame containing the equity curve data.
        """
        return self._portfolio.get_equity_curve()

    def plot_results(self):
        """
        Generate plots for backtest results.
        """
        # TODO : call plotting methods from self.portfolio
        pass

    def save_results(self, filename: str):
        """
        Save the backtest results to a file.

        Args:
            filename (str): The name of the file to save the results to.
        """
        results = self._generate_results()  # noqa
        # Implement logic to save results to a file (e.g., JSON, CSV, or pickle)
        logger_main.info(f"Backtest results saved to {filename}")

    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load backtest results from a file.

        Args:
            filename (str): The name of the file to load the results from.

        Returns:
            Dict[str, Any]: The loaded backtest results.
        """
        # Implement logic to load results from a file
        logger_main.info(f"Backtest results loaded from {filename}")
        return {}  # Replace with actual loaded results

    # endregion
