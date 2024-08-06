from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main


class Data:
    def __init__(self, symbol: str, max_length: int = 1000):
        self.symbol: str = symbol
        self.max_length: int = max_length
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
        for attr in ["open", "high", "low", "close", "volume"]:
            value = getattr(bar, attr)
            self._data[bar.timeframe][attr] = np.concatenate(
                ([value], self._data[bar.timeframe][attr][: self.max_length - 1])
            )

    def __getitem__(self, timeframe: Timeframe) -> Dict[str, np.ndarray]:
        # Ensure the timeframe exists in the data
        if timeframe not in self._data:
            self._data[timeframe] = {
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
                "volume": np.array([], dtype=float),
            }
        return self._data[timeframe]

    def get(
        self, timeframe: Timeframe, attr: str, default: Optional[float] = None
    ) -> np.ndarray:
        return self[timeframe].get(attr, default)

    @property
    def timeframes(self):
        return list(self._data.keys())


class DataTimeframe:
    def __init__(self, data: Data, timeframe: Timeframe):
        self.data = data
        self.timeframe = timeframe

    @property
    def open(self) -> np.ndarray:
        return self.data._arrays[self.timeframe]["open"]

    @property
    def high(self) -> np.ndarray:
        return self.data._arrays[self.timeframe]["high"]

    @property
    def low(self) -> np.ndarray:
        return self.data._arrays[self.timeframe]["low"]

    @property
    def close(self) -> np.ndarray:
        return self.data._arrays[self.timeframe]["close"]

    @property
    def volume(self) -> np.ndarray:
        return self.data._arrays[self.timeframe]["volume"]


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
        return f"BarManager(symbols={symbol_count}, total_bars={total_bars})"
