from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .data.dataloader import BaseDataLoader
from .data.dataview import DataView, OptimizedDataView
from .data.timeframe import Timeframe
from .log_config import logger_main


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

    def build_optimized_dataview(self):
        """
        Build an optimized view of the data for faster access during backtesting.

        This method aligns all data in the DataView and creates an OptimizedDataView instance.
        """
        self._dataview.align_all_data()
        self.optimized_dataview = OptimizedDataView(self._dataview)

    def add_strategy(self, strategy):
        """
        Add a trading strategy to the engine.

        Args:
            strategy: An instance of a trading strategy class.
        """
        self.strategies.append(strategy)
        logger_main.log_and_print(f"Added strategy: {strategy.__class__.__name__}")

    def initialize_portfolio(self, initial_capital: float):
        """
        Initialize the portfolio with a given amount of initial capital.

        Args:
            initial_capital (float): The initial amount of capital in the portfolio.
        """
        self.portfolio = Portfolio(initial_capital)
        logger_main.log_and_print(f"Initialized portfolio with ${initial_capital}")

    def run(self):
        """
        Run the backtest.

        This method executes the backtesting process, applying strategies to historical data
        and tracking portfolio performance.

        Raises:
            ValueError: If the portfolio is not initialized or no strategies are added.
        """
        if not self.optimized_dataview:
            self.build_optimized_dataview()

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
            self.process_timestamp(timestamp, data_point)

        self.generate_results()

    def process_timestamp(self, timestamp: pd.Timestamp, data_point: dict):
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
                    self.process_data_point(symbol, timeframe, timestamp, ohlcv_data)

    def process_data_point(
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

    def generate_results(self):
        """
        Generate and log the results of the backtest.

        This method calculates final portfolio performance and logs the results.
        """
        results = self.portfolio.get_results()
        logger_main.log_and_print("Backtest completed. Generating results...")
        print(results)
        # Add code to save or display results


class Portfolio:
    """
    A class representing the trading portfolio, tracking positions and performance.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize the Portfolio instance.

        Args:
            initial_capital (float): The initial amount of capital in the portfolio.
        """
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.history: List[Dict] = []

    def execute_order(
        self, action: str, symbol: str, price: float, timestamp: pd.Timestamp
    ):
        """
        Execute a trading order, updating the portfolio's positions and cash balance.

        Args:
            action (str): The action to take ('buy' or 'sell').
            symbol (str): The symbol to trade.
            price (float): The price at which to execute the trade.
            timestamp (pd.Timestamp): The timestamp of the trade.
        """
        # Implement order execution logic
        pass

    def update(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """
        Update the portfolio value based on current market prices.

        Args:
            timestamp (pd.Timestamp): The current timestamp.
            prices (Dict[str, float]): A dictionary of current prices for each symbol.
        """
        portfolio_value = self.cash + sum(
            self.positions.get(symbol, 0) * price for symbol, price in prices.items()
        )
        self.history.append(
            {
                "timestamp": timestamp,
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "positions": self.positions.copy(),
            }
        )

    def get_results(self):
        """
        Calculate and return portfolio performance metrics.

        Returns:
            Dict: A dictionary containing various performance metrics.
        """
        # Calculate and return portfolio performance metrics
        pass
