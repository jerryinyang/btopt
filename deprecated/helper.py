from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main


class Data:
    """
    A class to manage market data for a specific symbol across multiple timeframes.

    This class provides efficient storage and access to OHLCV (Open, High, Low, Close, Volume)
    data for different timeframes.

    Attributes:
        symbol (str): The market symbol (e.g., 'EURUSD', 'AAPL') this data represents.
        max_length (int): The maximum number of data points to store for each timeframe.
        _data (Dict[Timeframe, Dict[str, np.ndarray]]): The internal data storage structure.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the Data object.

        Args:
            symbol (str): The market symbol this data represents.
            max_length (int, optional): The maximum number of data points to store. Defaults to 1000.
        """
        self.symbol: str = symbol
        self._max_length: int = max_length
        self._data: Dict[Timeframe, Dict[str, np.ndarray]] = defaultdict(
            lambda: {
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
                "volume": np.array([], dtype=float),
            }
        )

    def add_bar(self, bar: Bar) -> None:
        """
        Add a new bar of data to the appropriate timeframe.

        Args:
            bar (Bar): The new bar of market data to add.
        """
        for attr in ["open", "high", "low", "close", "volume"]:
            value = getattr(bar, attr)
            self._data[bar.timeframe][attr] = np.concatenate(
                ([value], self._data[bar.timeframe][attr][: self.max_length - 1])
            )

    def __getitem__(self, timeframe: Timeframe) -> "DataTimeframe":
        """
        Access data for a specific timeframe.

        Args:
            timeframe (Timeframe): The timeframe to access data for.

        Returns:
            DataTimeframe: A DataTimeframe object providing access to the data for the specified timeframe.
        """
        # Ensure the timeframe exists in the data
        if timeframe not in self._data:
            self._data[timeframe] = {
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
                "volume": np.array([], dtype=float),
            }
        return DataTimeframe(self, timeframe)

    def get(
        self, timeframe: Timeframe, attr: str, default: Optional[float] = None
    ) -> np.ndarray:
        """
        Get data for a specific timeframe and attribute.

        Args:
            timeframe (Timeframe): The timeframe to get data for.
            attr (str): The attribute to get ('open', 'high', 'low', 'close', or 'volume').
            default (Optional[float], optional): Default value if data is not found. Defaults to None.

        Returns:
            np.ndarray: The requested data as a numpy array.
        """
        return self._data[timeframe].get(attr, default)

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            logger_main.log_and_raise(
                ValueError("max_length must be a positive integer.")
            )

        old_max_length = self._max_length
        self._max_length = value

        # Update all existing data arrays
        for timeframe_dict in self._data.values():
            for key, array in timeframe_dict.items():
                if value > old_max_length:
                    # If increasing, pad with NaN at the end
                    pad_width = value - len(array)
                    timeframe_dict[key] = np.pad(
                        array, (0, pad_width), constant_values=np.nan
                    )
                else:
                    # If decreasing, trim the existing array
                    timeframe_dict[key] = array[:value]

        logger_main.info(f"max_length updated from {old_max_length} to {value}")

    @property
    def timeframes(self) -> List[Timeframe]:
        """Get a list of all available timeframes."""
        return list(self._data.keys())

    @property
    def open(self) -> Dict[Timeframe, np.ndarray]:
        """Get open price data for all timeframes."""
        return {tf: data["open"] for tf, data in self._data.items()}

    @property
    def high(self) -> Dict[Timeframe, np.ndarray]:
        """Get high price data for all timeframes."""
        return {tf: data["high"] for tf, data in self._data.items()}

    @property
    def low(self) -> Dict[Timeframe, np.ndarray]:
        """Get low price data for all timeframes."""
        return {tf: data["low"] for tf, data in self._data.items()}

    @property
    def close(self) -> Dict[Timeframe, np.ndarray]:
        """Get close price data for all timeframes."""
        return {tf: data["close"] for tf, data in self._data.items()}

    @property
    def volume(self) -> Dict[Timeframe, np.ndarray]:
        """Get volume data for all timeframes."""
        return {tf: data["volume"] for tf, data in self._data.items()}


class DataTimeframe:
    """
    A class to provide convenient access to market data for a specific timeframe.

    This class acts as a view into the Data object, providing easy access to OHLCV data
    for a particular timeframe.

    Attributes:
        _data (Data): The parent Data object this view is associated with.
        _timeframe (Timeframe): The specific timeframe this view represents.
    """

    def __init__(self, data: Data, timeframe: Timeframe):
        """
        Initialize the DataTimeframe object.

        Args:
            data (Data): The parent Data object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        self._data = data
        self._timeframe = timeframe

    @property
    def open(self) -> np.ndarray:
        """Get the open price data for this timeframe."""
        return self._data._data[self._timeframe]["open"]

    @property
    def high(self) -> np.ndarray:
        """Get the high price data for this timeframe."""
        return self._data._data[self._timeframe]["high"]

    @property
    def low(self) -> np.ndarray:
        """Get the low price data for this timeframe."""
        return self._data._data[self._timeframe]["low"]

    @property
    def close(self) -> np.ndarray:
        """Get the close price data for this timeframe."""
        return self._data._data[self._timeframe]["close"]

    @property
    def volume(self) -> np.ndarray:
        """Get the volume data for this timeframe."""
        return self._data._data[self._timeframe]["volume"]

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Access OHLCV data using dictionary-style key access.

        Args:
            key (str): The data to access ('open', 'high', 'low', 'close', or 'volume').

        Returns:
            np.ndarray: The requested data as a numpy array.

        Raises:
            KeyError: If an invalid key is provided.
        """
        if key not in ["open", "high", "low", "close", "volume"]:
            raise KeyError(
                f"Invalid key: {key}. Must be one of 'open', 'high', 'low', 'close', or 'volume'."
            )

        data = self._data._data[self._timeframe][key]

        if data.size == 0:
            logger_main.warning(
                f"No data available for {key} in timeframe {self._timeframe}"
            )
            return np.array([])

        return data

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the OHLCV data for this timeframe to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data.
        """
        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            }
        )


class BarManager:
    """
    A class to manage and provide access to Bar objects for multiple symbols and timeframes.

    This class acts as a container for Bar objects, organizing them by symbol and timeframe.
    It provides methods for adding new bars, accessing historical bars, and managing the
    data structure efficiently.

    Attributes:
        max_bars (int): Maximum number of bars to store for each symbol-timeframe combination.
        _bars (Dict[str, Dict[Timeframe, deque]]): The internal storage structure for bars.
    """

    def __init__(self, max_bars: int = 500):
        """
        Initialize the BarManager.

        Args:
            max_bars (int, optional): Maximum number of bars to store for each
                                      symbol-timeframe combination. Defaults to 500.
        """
        self._max_bars: int = max_bars
        self._bars: Dict[str, Dict[Timeframe, deque]] = {}

    @property
    def max_bars(self) -> int:
        return self._max_bars

    @max_bars.setter
    def max_bars(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            logger_main.log_and_raise(
                ValueError("max_bars must be a positive integer.")
            )

        old_max_bars = self._max_bars
        self._max_bars = value

        # Update all existing deques
        for symbol_dict in self._bars.values():
            for timeframe, timeframe_deque in symbol_dict.items():
                if value > old_max_bars:
                    # If increasing, create a new deque with larger maxlen
                    new_deque = deque(timeframe_deque, maxlen=value)
                else:
                    # If decreasing, trim the existing deque
                    new_deque = deque(list(timeframe_deque)[-value:], maxlen=value)
                symbol_dict[timeframe] = new_deque

        logger_main.info(f"max_bars updated from {old_max_bars} to {value}")

    def add_bar(self, bar: Bar) -> None:
        """
        Add a new Bar object to the manager.

        This method adds the new bar to the appropriate symbol and timeframe queue.
        If the symbol or timeframe doesn't exist, it creates a new queue.

        Args:
            bar (Bar): The new Bar object to add.
        """
        symbol = bar.ticker
        timeframe = bar.timeframe

        # Initialize nested dictionaries if they don't exist
        if symbol not in self._bars:
            self._bars[symbol] = {}
        if timeframe not in self._bars[symbol]:
            self._bars[symbol][timeframe] = deque(maxlen=self._max_bars)

        # Add the new bar
        self._bars[symbol][timeframe].appendleft(bar)

    def get_bar(
        self, symbol: str, timeframe: Timeframe, index: int = 0
    ) -> Optional[Bar]:
        """
        Retrieve a specific Bar object.

        Args:
            symbol (str): The symbol of the desired bar.
            timeframe (Timeframe): The timeframe of the desired bar.
            index (int, optional): The index of the bar to retrieve. 0 is the most recent,
                                   -1 is the previous, and so on. Defaults to 0.

        Returns:
            Optional[Bar]: The requested Bar object, or None if not found.

        Raises:
            IndexError: If the requested index is out of range.

        Note:
            BarManager uses negative indexing (0 for most recent, -1 for previous),
            unlike the Data class which uses positive indexing for OHLCV data.
        """
        try:
            return self._bars[symbol][timeframe][index]
        except KeyError:
            return None
        except IndexError:
            logger_main.log_and_raise(
                IndexError(f"Bar index {index} out of range for {symbol} {timeframe}")
            )

    def get_latest_bars(
        self, symbol: str, timeframe: Timeframe, n: int = 1
    ) -> List[Bar]:
        """
        Retrieve the latest n Bar objects for a given symbol and timeframe.

        Args:
            symbol (str): The symbol to retrieve bars for.
            timeframe (Timeframe): The timeframe to retrieve bars for.
            n (int, optional): The number of bars to retrieve. Defaults to 1.

        Returns:
            List[Bar]: A list of the latest n Bar objects.
        """
        try:
            return list(self._bars[symbol][timeframe])[:n]
        except KeyError:
            return []

    def get_dataframe(
        self, symbol: str, timeframe: Timeframe, n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert stored Bar objects to a pandas DataFrame.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.
            n (Optional[int], optional): The number of latest bars to include.
                                         If None, includes all stored bars. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the bar data.
        """
        try:
            bars = (
                list(self._bars[symbol][timeframe])[:n]
                if n is not None
                else self._bars[symbol][timeframe]
            )
            data = [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df
        except KeyError:
            return pd.DataFrame()

    def get_symbols(self) -> List[str]:
        """
        Get a list of all symbols in the BarManager.

        Returns:
            List[str]: A list of all symbols.
        """
        return list(self._bars.keys())

    def get_timeframes(self, symbol: str) -> List[Timeframe]:
        """
        Get a list of all timeframes for a given symbol.

        Args:
            symbol (str): The symbol to get timeframes for.

        Returns:
            List[Timeframe]: A list of all timeframes for the given symbol.
        """
        return list(self._bars.get(symbol, {}).keys())

    def __len__(self) -> int:
        """
        Get the total number of bars stored across all symbols and timeframes.

        Returns:
            int: The total number of bars.
        """
        return sum(
            len(timeframe_deque)
            for symbol_dict in self._bars.values()
            for timeframe_deque in symbol_dict.values()
        )

    def __repr__(self) -> str:
        """
        Get a string representation of the BarManager.

        Returns:
            str: A string representation of the BarManager.
        """
        symbol_count = len(self.get_symbols())
        total_bars = len(self)
        return f"BarManager(symbols={symbol_count}, total_bars={total_bars})"
