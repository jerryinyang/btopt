from typing import Union

import numpy as np
import pandas as pd

from .data.dataloader import BaseDataLoader
from .data.dataview import DataView, DataViewNumpy
from .data.timeframe import Timeframe
from .log_config import logger_main
from .portfolio import Portfolio


class Engine:
    """
    The main backtesting engine class that coordinates data management, strategy execution, and portfolio tracking.

    This class serves as the central hub for the backtesting process, managing data loading,
    optimization, strategy application, and result generation.
    """

    def __init__(self):
        """
        Initialize the Engine instance.

        Sets up the basic DataView and prepares containers for the optimized view, strategies, and portfolio.
        """
        self._dataview = DataView()
        self.optimized_dataview = None
        self.strategies = []
        self.portfolio = None

    # Backtest Operation
    def run(self):
        """
        Run the backtest.

        This method executes the backtesting process, applying strategies to historical data
        and tracking portfolio performance.

        Raises:
            ValueError: If the portfolio is not initialized or no strategies are added.
        """
        if not self.optimized_dataview:
            self._init_optimized_dataview()

        if not self.portfolio:
            logger_main.log_and_raise(
                ValueError(
                    "Portfolio not initialized. Call initialize_portfolio() first."
                )
            )

        if not self.strategies:
            logger_main.log_and_raise(
                ValueError(
                    "No strategies added. Add at least one strategy before running."
                )
            )

        for timestamp, data_point in self.optimized_dataview:
            self._process_timestamp(timestamp, data_point)

        self._generate_results()

    def add_strategy(self, strategy):
        """
        Add a trading strategy to the engine.

        Args:
            strategy: An instance of a trading strategy class.
        """
        self.strategies.append(strategy)
        logger_main.log_and_print(f"Added strategy: {strategy.__class__.__name__}")

    def _init_portfolio(self, initial_capital: float):
        """
        Initialize the portfolio with a given amount of initial capital.

        Args:
            initial_capital (float): The initial amount of capital in the portfolio.
        """
        self.portfolio = Portfolio(initial_capital)
        logger_main.log_and_print(f"Initialized portfolio with ${initial_capital}")

    def _init_optimized_dataview(self):
        """
        Build an optimized view of the data for faster access during backtesting.

        This method aligns all data in the DataView and creates an DataViewNumpy instance.
        """
        self._dataview.align_all_data()
        self.optimized_dataview = DataViewNumpy(self._dataview)

    def _process_timestamp(self, timestamp: pd.Timestamp, data_point: dict):
        """
        Process data for a specific timestamp across all symbols and timeframes.

        Args:
            timestamp (pd.Timestamp): The timestamp being processed.
            data_point (dict): The data point containing information for all symbols and timeframes.
        """
        for symbol in self.optimized_dataview.symbols:
            for timeframe in self.optimized_dataview.timeframes:
                if self.optimized_dataview.is_original_data_point(
                    symbol, timeframe, timestamp
                ):
                    ohlcv_data = self.optimized_dataview.get_data_point_by_keys(
                        symbol, timeframe, timestamp
                    )
                    self._process_data_point(symbol, timeframe, timestamp, ohlcv_data)

    def _process_data_point(
        self,
        symbol: str,
        timeframe: Timeframe,
        timestamp: pd.Timestamp,
        ohlcv_data: np.ndarray,
    ):
        """
        Process a single data point, applying all strategies and updating the portfolio.

        Args:
            symbol (str): The symbol being processed.
            timeframe (Timeframe): The timeframe of the data point.
            timestamp (pd.Timestamp): The timestamp of the data point.
            ohlcv_data (np.ndarray): The OHLCV data for the given symbol and timestamp.
        """
        for strategy in self.strategies:
            action = strategy.generate_signal(symbol, timeframe, timestamp, ohlcv_data)
            if action:
                self.portfolio.execute_order(
                    action, symbol, ohlcv_data[3], timestamp
                )  # Assuming close price is at index 3

        self.portfolio.update(
            timestamp,
            {symbol: ohlcv_data[3] for symbol in self.optimized_dataview.symbols},
        )

    def _generate_results(self):
        """
        Generate and log the results of the backtest.

        This method calculates final portfolio performance and logs the results.
        """
        results = self.portfolio.get_results()
        logger_main.log_and_print("Backtest completed. Generating results...")
        print(results)
        # Add code to save or display results

    # Data Processing
    def add_data(self, dataloader: BaseDataLoader):
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
    ):
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

    # Order Management
    def buy(self, order_details):
        """
        Place a new buy order in the system.

        Args:
            order_details (dict): Details of the order including symbol, type, size, etc.

        Returns:
            int: The ID of the newly created order.
        """
        pass

    def sell(self, order_details):
        """
        Place a new buy order in the system.

        Args:
            order_details (dict): Details of the order including symbol, type, size, etc.

        Returns:
            int: The ID of the newly created order.
        """
        pass

    def _place_order(self, order_details):
        """
        Place a new order in the system.

        Args:
            order_details (dict): Details of the order including symbol, type, size, etc.

        Returns:
            int: The ID of the newly created order.
        """
        pass

    def _cancel_order(self, order_id):
        """
        Cancel an existing order.

        Args:
            order_id (int): The ID of the order to cancel.

        Returns:
            bool: True if the order was successfully canceled, False otherwise.
        """
        pass

    def _modify_order(self, order_id, new_details):
        """
        Modify an existing order.

        Args:
            order_id (int): The ID of the order to modify.
            new_details (dict): New details for the order.

        Returns:
            bool: True if the order was successfully modified, False otherwise.
        """
        pass

    def _execute_orders(self, current_bar):
        """
        Check and execute pending orders based on the current market data.

        Args:
            current_bar (Bar): The current price bar.

        Returns:
            list: A list of executed order IDs.
        """
        pass

    # Trade Management
    def _create_trade(self, order_id):
        """
        Create a new trade based on an executed order.

        Args:
            order_id (int): The ID of the executed order.

        Returns:
            int: The ID of the newly created trade.
        """
        pass

    def _close_trade(self, trade_id, exit_order_id):
        """
        Close an existing trade.

        Args:
            trade_id (int): The ID of the trade to close.
            exit_order_id (int): The ID of the exit order.

        Returns:
            bool: True if the trade was successfully closed, False otherwise.
        """
        pass

    def _update_trades(self, current_bar):
        """
        Update all open trades based on the current market data.

        Args:
            current_bar (Bar): The current price bar.
        """
        pass

    # Getters
    def get_order_status(self, order_id):
        """
        Get the current status of an order.

        Args:
            order_id (int): The ID of the order.

        Returns:
            str: The status of the order.
        """
        pass

    def get_trade_status(self, trade_id):
        """
        Get the current status of a trade.

        Args:
            trade_id (int): The ID of the trade.

        Returns:
            str: The status of the trade.
        """
        pass

    def get_portfolio_status(self):
        """
        Get the current status of the portfolio.

        Returns:
            dict: A dictionary containing portfolio metrics.
        """
        pass
        pass
