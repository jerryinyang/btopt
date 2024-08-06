import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main
from ..order import Order
from ..parameters import Parameters
from ..trade import Trade
from .helper import BarManager, Data


class StrategyError(Exception):
    pass


def generate_unique_id() -> str:
    """Generate a unique identifier for the strategy."""
    return str(uuid.uuid4())


class Strategy(ABC):
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

    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        primary_timeframe: Optional[Timeframe] = None,
    ) -> None:
        """
        Initialize the Strategy instance.

        Args:
            name (str): The name of the strategy.
            parameters (Optional[Dict[str, Any]]): Initial strategy parameters. Defaults to None.
            primary_timeframe (Optional[Timeframe]): The primary timeframe for the strategy. Defaults to None.
        """
        self.name: str = name
        self._parameters: Parameters = Parameters(parameters or {})
        self.primary_timeframe: Optional[Timeframe] = primary_timeframe
        self._datas: Dict[str, Dict[Timeframe, Data]] = defaultdict(dict)
        self._bar_manager: BarManager = BarManager()
        self._initialized: bool = False
        self._primary_symbol: Optional[str] = None
        self._engine = None
        self._strategy_timeframes: Dict[Timeframe, List[str]] = {}
        self._id: str = generate_unique_id()
        self._positions: Dict[str, float] = {}
        self._pending_orders: List[Order] = []

    def initialize(
        self,
        symbols: List[str],
        timeframes: List[Timeframe],
        default_timeframe: Timeframe,
    ) -> None:
        """
        Initialize the strategy with symbols and timeframes.

        Args:
            symbols (List[str]): List of symbols the strategy will trade.
            timeframes (List[Timeframe]): List of timeframes the strategy will use.
            default_timeframe (Timeframe): Default timeframe to use if primary_timeframe is not set.

        Raises:
            StrategyError: If the strategy is already initialized.
        """
        if self._initialized:
            raise StrategyError("Strategy is already initialized.")

        if self.primary_timeframe is None:
            self.primary_timeframe = default_timeframe
        elif self.primary_timeframe not in timeframes:
            logger_main.warning(
                f"Specified primary timeframe {self.primary_timeframe} not in available timeframes. Using default: {default_timeframe}"
            )
            self.primary_timeframe = default_timeframe

        for symbol in symbols:
            for timeframe in timeframes:
                self._datas[symbol][timeframe] = Data(symbol, timeframe)
            self._positions[symbol] = 0.0

        self._primary_symbol = symbols[0] if symbols else None
        self._initialized = True
        logger_main.info(f"Initialized strategy: {self.name}")
        logger_main.info(f"Symbols: {symbols}")
        logger_main.info(f"Timeframes: {timeframes}")
        logger_main.info(f"Primary timeframe: {self.primary_timeframe}")
        logger_main.info(f"Primary symbol: {self._primary_symbol}")

    @property
    def data(self) -> Data:
        """
        Get the primary data stream for the strategy.

        Returns:
            Data: The primary Data object for the strategy.

        Raises:
            StrategyError: If the strategy hasn't been initialized or if there's no primary symbol.
        """
        if not self._initialized:
            raise StrategyError("Strategy has not been initialized.")
        if self._primary_symbol is None:
            raise StrategyError("No primary symbol set for the strategy.")
        try:
            return self._datas[self._primary_symbol][self.primary_timeframe]
        except KeyError:
            raise StrategyError(
                f"Data for primary symbol {self._primary_symbol} and timeframe {self.primary_timeframe} not found."
            )

    @property
    def parameters(self) -> Parameters:
        """
        Get the strategy parameters.

        Returns:
            Parameters: The strategy parameters object.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Set new strategy parameters.

        Args:
            new_parameters (Dict[str, Any]): A dictionary of new parameter values.
        """
        self._parameters = Parameters(new_parameters)
        logger_main.info(
            f"Updated parameters for strategy {self.name}: {new_parameters}"
        )

    @abstractmethod
    def on_bar(self, bar: Bar) -> None:
        """
        Handle the arrival of a new price bar.

        This method is the primary decision-making point for the strategy.
        It should be implemented by concrete strategy classes to define the strategy's logic.

        Args:
            bar (Bar): The new price bar data.
        """
        pass

    def process_bar(self, bar: Bar) -> None:
        """
        Process a new bar and update the strategy's state.

        This method is called by the Engine for each new bar. It updates the strategy's
        internal data structures and calls the on_bar method for decision making.

        Args:
            bar (Bar): The new price bar data.
        """
        symbol, timeframe = bar.ticker, bar.timeframe
        self._datas[symbol][timeframe].add_bar(bar)
        self._bar_manager.add_bar(bar)
        self.on_bar(bar)

    def on_order_update(self, order: Order) -> None:
        """
        Handle updates to orders created by this strategy.

        Args:
            order (Order): The updated Order object.
        """
        # Update the strategy's state based on the order update
        if order.status == Order.Status.FILLED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
            self._update_position(order)
        elif order.status == Order.Status.CANCELED:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]

        # Implement any strategy-specific logic for handling order updates
        self._handle_order_update(order)

    def on_trade_update(self, trade: Trade) -> None:
        """
        Handle updates to trades associated with this strategy.

        Args:
            trade (Trade): The updated Trade object.
        """
        # Update the strategy's state based on the trade update
        if trade.status == Trade.Status.CLOSED:
            self._update_position(trade)

        # Implement any strategy-specific logic for handling trade updates
        self._handle_trade_update(trade)

    @abstractmethod
    def _handle_order_update(self, order: Order) -> None:
        """
        Handle strategy-specific logic for order updates.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to order updates.

        Args:
            order (Order): The updated Order object.
        """
        pass

    @abstractmethod
    def _handle_trade_update(self, trade: Trade) -> None:
        """
        Handle strategy-specific logic for trade updates.

        This method should be implemented by concrete strategy classes to define
        how the strategy responds to trade updates.

        Args:
            trade (Trade): The updated Trade object.
        """
        pass

    def _update_position(self, transaction: Union[Order, Trade]) -> None:
        """
        Update the strategy's position based on an order or trade.

        Args:
            transaction (Union[Order, Trade]): The Order or Trade object to update the position from.
        """
        symbol = transaction.ticker
        if isinstance(transaction, Order):
            size = transaction.details.size
        else:  # Trade
            size = transaction.initial_size

        if transaction.direction == Order.Direction.LONG:
            self._positions[symbol] += size
        else:  # SHORT
            self._positions[symbol] -= size

    def buy(
        self, symbol: str, size: float, price: Optional[float] = None, **kwargs: Any
    ) -> Optional[Order]:
        """
        Create a buy order.

        Args:
            symbol (str): The symbol to buy.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            **kwargs: Additional order parameters.

        Returns:
            Optional[Order]: The created Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        order = self._engine.create_order(
            self, symbol, Order.Direction.LONG, size, price, **kwargs
        )
        if order:
            self._pending_orders.append(order)
        return order

    def sell(
        self, symbol: str, size: float, price: Optional[float] = None, **kwargs: Any
    ) -> Optional[Order]:
        """
        Create a sell order.

        Args:
            symbol (str): The symbol to sell.
            size (float): The size of the order.
            price (Optional[float]): The price for limit orders. If None, a market order is created.
            **kwargs: Additional order parameters.

        Returns:
            Optional[Order]: The created Order object, or None if the order creation failed.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        order = self._engine.create_order(
            self, symbol, Order.Direction.SHORT, size, price, **kwargs
        )
        if order:
            self._pending_orders.append(order)
        return order

    def cancel(self, order: Order) -> bool:
        """
        Cancel an existing order.

        Args:
            order (Order): The order to cancel.

        Returns:
            bool: True if the order was successfully cancelled, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        success = self._engine.cancel_order(self, order)
        if success:
            self._pending_orders = [o for o in self._pending_orders if o.id != order.id]
        return success

    def close(self, symbol: Optional[str] = None) -> bool:
        """
        Close all positions for this strategy, or for a specific symbol if provided.

        Args:
            symbol (Optional[str]): The symbol to close positions for. If None, close all positions.

        Returns:
            bool: True if the closing operation was successful, False otherwise.

        Raises:
            StrategyError: If the strategy is not connected to an engine.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")
        success = self._engine.close_positions(self, symbol)
        if success and symbol:
            self._positions[symbol] = 0.0
        elif success:
            self._positions = {s: 0.0 for s in self._positions}
        return success

    def calculate_position_size(
        self, symbol: str, risk_percent: float, stop_loss: float
    ) -> float:
        """
        Calculate the position size based on risk percentage and stop loss.

        Args:
            symbol (str): The symbol to trade.
            risk_percent (float): The percentage of account to risk on this trade.
            stop_loss (float): The stop loss price.

        Returns:
            float: The calculated position size.

        Raises:
            StrategyError: If the strategy is not connected to an engine or if the risk per share is zero.
        """
        if self._engine is None:
            raise StrategyError("Strategy is not connected to an engine.")

        account_value = self._engine.get_account_value()
        risk_amount = account_value * (risk_percent / 100)
        current_price = self._datas[symbol][self.primary_timeframe].close[-1]
        risk_per_share = abs(current_price - stop_loss)

        if risk_per_share == 0:
            raise StrategyError(
                "Risk per share is zero. Cannot calculate position size."
            )

        position_size = risk_amount / risk_per_share
        return position_size

    def get_current_position(self, symbol: str) -> float:
        """
        Get the current position size for a given symbol.

        Args:
            symbol (str): The symbol to check.

        Returns:
            float: The current position size (positive for long, negative for short, 0 for no position).
        """
        return self._positions.get(symbol, 0.0)

    def __getitem__(self, key: Union[str, Tuple[str, Timeframe]]) -> Data:
        """
        Allow easy access to data streams.

        Args:
            key (Union[str, Tuple[str, Timeframe]]):
                If str, assumes primary timeframe and returns data for that symbol.
                If tuple, returns data for the specified (symbol, timeframe) pair.

        Returns:
            Data: The requested Data object.

        Raises:
            KeyError: If the requested symbol or timeframe is not found.
        """
        if isinstance(key, str):
            return self._datas[key][self.primary_timeframe]
        elif isinstance(key, tuple) and len(key) == 2:
            symbol, timeframe = key
            return self._datas[symbol][timeframe]
        else:
            raise KeyError(f"Invalid key: {key}")

    def get_data(self, symbol: str, timeframe: Optional[Timeframe] = None) -> Data:
        """
        Retrieve a specific data stream.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Optional[Timeframe]): The timeframe to retrieve data for.
                                             If None, uses the primary timeframe.

        Returns:
            Data: The requested Data object.

        Raises:
            KeyError: If the requested symbol or timeframe is not found.
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
        return self._datas[symbol][timeframe]

    def get_pending_orders(self) -> List[Order]:
        """
        Get all pending orders for this strategy.

        Returns:
            List[Order]: A list of all pending orders.
        """
        return self._pending_orders

    def get_positions(self) -> Dict[str, float]:
        """
        Get all current positions for this strategy.

        Returns:
            Dict[str, float]: A dictionary of current positions, keyed by symbol.
        """
        return self._positions.copy()

    def set_engine(self, engine: Any) -> None:
        """
        Set the engine for this strategy.

        Args:
            engine (Any): The engine object to set.
        """
        self._engine = engine

    def __repr__(self) -> str:
        """
        Return a string representation of the Strategy.

        Returns:
            str: A string representation of the Strategy.
        """
        return (
            f"Strategy(name={self.name}, id={self._id}, "
            f"symbols={list(self._datas.keys())}, "
            f"timeframes={list(next(iter(self._datas.values())).keys())}, "
            f"positions={self._positions})"
        )
