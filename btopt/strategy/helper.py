from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe


class Data:
    """
    A class to manage and provide access to financial data for a specific symbol and timeframe.

    This class stores historical bar data and provides efficient access to OHLCV
    (Open, High, Low, Close, Volume) information. It uses a combination of a deque
    for storing Bar objects and numpy arrays for fast numerical operations on OHLCV data.

    Attributes:
        symbol (str): The financial instrument symbol (e.g., 'AAPL' for Apple Inc.).
        timeframe (Timeframe): The timeframe of the data (e.g., '1m' for 1-minute bars).
        max_length (int): Maximum number of historical bars to store.
    """

    def __init__(self, symbol: str, timeframe: Timeframe, max_length: int = 1000):
        """
        Initialize the Data object.

        Args:
            symbol (str): The financial instrument symbol.
            timeframe (Timeframe): The timeframe of the data.
            max_length (int, optional): Maximum number of historical bars to store. Defaults to 1000.
        """
        self.symbol: str = symbol
        self.timeframe: Timeframe = timeframe
        self.max_length: int = max_length
        self._bars: deque = deque(maxlen=max_length)

        # Initialize numpy arrays for OHLCV data
        self._open: np.ndarray = np.array([], dtype=float)
        self._high: np.ndarray = np.array([], dtype=float)
        self._low: np.ndarray = np.array([], dtype=float)
        self._close: np.ndarray = np.array([], dtype=float)
        self._volume: np.ndarray = np.array([], dtype=float)

    def __getitem__(self, index: int) -> Bar:
        """
        Allow indexing to access historical bars.

        Args:
            index (int): The index of the bar to retrieve. 0 is the most recent bar,
                         -1 is the previous bar, and so on.

        Returns:
            Bar: The requested historical bar.

        Raises:
            IndexError: If the requested index is out of range.
        """
        if abs(index) >= len(self._bars):
            raise IndexError("Data index out of range")
        return self._bars[index]

    def add_bar(self, bar: Bar) -> None:
        """
        Add a new bar to the data and update the OHLCV arrays.

        This method adds the new bar to the front of the deque and updates
        the numpy arrays with the new data.

        Args:
            bar (Bar): The new bar to add to the data.
        """
        self._bars.appendleft(bar)
        self._update_arrays(bar)

    def _update_arrays(self, bar: Bar) -> None:
        """
        Update the OHLCV numpy arrays with the new bar data.

        This private method is called by add_bar to keep the numpy arrays
        in sync with the _bars deque.

        Args:
            bar (Bar): The new bar containing the data to add to the arrays.
        """
        for attr in ["open", "high", "low", "close", "volume"]:
            arr = getattr(self, f"_{attr}")
            value = getattr(bar, attr)

            # Insert the new value at the beginning and trim to max_length
            setattr(
                self, f"_{attr}", np.concatenate(([value], arr[: self.max_length - 1]))
            )

    @property
    def open(self) -> np.ndarray:
        """np.ndarray: Array of historical open prices."""
        return self._open

    @property
    def high(self) -> np.ndarray:
        """np.ndarray: Array of historical high prices."""
        return self._high

    @property
    def low(self) -> np.ndarray:
        """np.ndarray: Array of historical low prices."""
        return self._low

    @property
    def close(self) -> np.ndarray:
        """np.ndarray: Array of historical close prices."""
        return self._close

    @property
    def volume(self) -> np.ndarray:
        """np.ndarray: Array of historical volume data."""
        return self._volume

    def fetch_data(self, by: str = "close", size: int = 1) -> np.ndarray:
        """
        Fetch a specified amount of historical data.

        Args:
            by (str, optional): The type of data to fetch. Must be one of
                                'open', 'high', 'low', 'close', or 'volume'.
                                Defaults to 'close'.
            size (int, optional): The number of historical data points to fetch.
                                  Defaults to 1.

        Returns:
            np.ndarray: Array of requested historical data.

        Raises:
            ValueError: If an invalid 'by' parameter is provided.
        """
        if by not in ["open", "high", "low", "close", "volume"]:
            raise ValueError(
                "Invalid 'by' parameter. Must be one of 'open', 'high', 'low', 'close', or 'volume'."
            )
        return getattr(self, by)[:size]

    def __len__(self) -> int:
        """
        Get the number of bars currently stored in the data.

        Returns:
            int: The number of bars in the data.
        """
        return len(self._bars)

    def __repr__(self) -> str:
        """
        Get a string representation of the Data object.

        Returns:
            str: A string representation of the Data object.
        """
        return (
            f"Data(symbol={self.symbol}, timeframe={self.timeframe}, bars={len(self)})"
        )


class BarManager:
    """
    A class to manage and provide access to Bar objects for multiple symbols and timeframes.

    This class acts as a container for Bar objects, organizing them by symbol and timeframe.
    It provides methods for adding new bars, accessing historical bars, and managing the
    data structure efficiently.

    Attributes:
        max_bars (int): Maximum number of bars to store for each symbol-timeframe combination.
    """

    def __init__(self, max_bars: int = 1000):
        """
        Initialize the BarManager.

        Args:
            max_bars (int, optional): Maximum number of bars to store for each
                                      symbol-timeframe combination. Defaults to 1000.
        """
        self.max_bars: int = max_bars
        self._bars: Dict[str, Dict[Timeframe, deque]] = {}

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
            self._bars[symbol][timeframe] = deque(maxlen=self.max_bars)

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
            raise IndexError(f"Bar index {index} out of range for {symbol} {timeframe}")

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
