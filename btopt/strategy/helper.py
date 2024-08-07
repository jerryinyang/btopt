from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..log_config import logger_main


class Data:
    """
    A class to manage market data for a specific symbol across multiple timeframes.

    This class provides efficient storage and access to OHLCV (Open, High, Low, Close, Volume)
    data for different timeframes, while maintaining compatibility with Bar objects.

    Attributes:
        symbol (str): The market symbol (e.g., 'EURUSD', 'AAPL') this data represents.
        _max_length (int): The maximum number of data points to store for each timeframe.
        _data (Dict[Timeframe, Dict[str, np.ndarray]]): The internal data storage structure.
        _timestamps (Dict[Timeframe, np.ndarray]): Timestamps for each timeframe.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the Data object.

        Args:
            symbol (str): The market symbol this data represents.
            max_length (int, optional): The maximum number of data points to store. Defaults to 500.
        """
        self.symbol: str = symbol
        self._max_length: int = max_length
        self._data: Dict[Timeframe, Dict[str, np.ndarray]] = defaultdict(
            self._create_empty_timeframe_data
        )
        self._timestamps: Dict[Timeframe, np.ndarray] = defaultdict(
            lambda: np.array([], dtype="datetime64[ns]")
        )

    @staticmethod
    def _create_empty_timeframe_data() -> Dict[str, np.ndarray]:
        """
        Create an empty data structure for a new timeframe.

        Returns:
            Dict[str, np.ndarray]: A dictionary with empty numpy arrays for OHLCV data.
        """
        return {
            "open": np.array([], dtype=float),
            "high": np.array([], dtype=float),
            "low": np.array([], dtype=float),
            "close": np.array([], dtype=float),
            "volume": np.array([], dtype=int),
        }

    def add_bar(self, bar: Bar) -> None:
        """
        Add a new bar of data to the appropriate timeframe.

        Args:
            bar (Bar): The new bar of market data to add.
        """
        timeframe = bar.timeframe
        for attr in ["open", "high", "low", "close"]:
            value = float(getattr(bar, attr))
            self._data[timeframe][attr] = np.concatenate(
                ([value], self._data[timeframe][attr][: self.max_length - 1])
            )
        self._data[timeframe]["volume"] = np.concatenate(
            ([bar.volume], self._data[timeframe]["volume"][: self.max_length - 1])
        )
        self._timestamps[timeframe] = np.concatenate(
            (
                [np.datetime64(bar.timestamp)],
                self._timestamps[timeframe][: self.max_length - 1],
            )
        )

    def get(
        self,
        timeframe: Optional[Timeframe] = None,
        index: int = 0,
        size: int = 1,
        value: Optional[str] = None,
    ) -> Union[
        Optional[Bar],
        List[Bar],
        Optional[Union[Decimal, int]],
        List[Union[Decimal, int]],
    ]:
        """
        Get Bar object(s) or specific value(s) for a given timeframe and index.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe. Defaults to None.
            index (int, optional): The starting index of the bar to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of bars to retrieve. Defaults to 1.
            value (Optional[str], optional): If specified, returns only this attribute of the Bar(s).
                                            Must be one of 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'timeframe', 'ticker', or 'index'.
                                            Defaults to None (returns full Bar object(s)).

        Returns:
            Union[Optional[Bar], List[Bar], Optional[Union[Decimal, int]], List[Union[Decimal, int]]]:
                - If value is None:
                    - If size is 1, returns a single Bar object or None if not available.
                    - If size > 1, returns a list of Bar objects (may be shorter than size if not enough data is available).
                - If value is specified:
                    - If size is 1, returns the specified value (Decimal for prices, int for volume, or appropriate type for other attributes) or None if not available.
                    - If size > 1, returns a list of the specified values.

        Raises:
            ValueError: If an invalid value is specified.
        """
        if timeframe is None:
            timeframe = self.primary_timeframe

        if timeframe not in self._data:
            return None if size == 1 else []

        valid_values = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp",
            "timeframe",
            "ticker",
            "index",
        }
        if value and value not in valid_values:
            raise ValueError(
                f"Invalid value: {value}. Must be one of {', '.join(valid_values)}"
            )

        result = []
        for i in range(index, min(index + size, len(self._data[timeframe]["open"]))):
            bar = Bar(
                open=Decimal(str(self._data[timeframe]["open"][i])),
                high=Decimal(str(self._data[timeframe]["high"][i])),
                low=Decimal(str(self._data[timeframe]["low"][i])),
                close=Decimal(str(self._data[timeframe]["close"][i])),
                volume=int(self._data[timeframe]["volume"][i]),
                timestamp=self._timestamps[timeframe][i].astype(datetime),
                timeframe=timeframe,
                ticker=self.symbol,
                index=i,
            )

            if value:
                result.append(getattr(bar, value))
            else:
                result.append(bar)

        if size == 1:
            return result[0] if result else None
        return result

    def __getitem__(self, key: Union[int, Timeframe]) -> "DataTimeframe":
        """
        Access data for a specific timeframe.

        Args:
            key (Union[int, Timeframe]):
                - If Timeframe: The specific timeframe to access data for.
                - If int: The index of the timeframe to access (0 for the primary/least timeframe).

        Returns:
            DataTimeframe: A DataTimeframe object providing access to the data for the specified timeframe.

        Raises:
            IndexError: If the integer index is out of range of available timeframes.
            TypeError: If the key is neither an integer nor a Timeframe.
        """
        if isinstance(key, int):
            timeframes = sorted(self.timeframes)
            if 0 <= key < len(timeframes):
                return DataTimeframe(self, timeframes[key])
            else:
                raise IndexError(
                    f"Timeframe index {key} is out of range. Available indices: 0 to {len(timeframes) - 1}"
                )
        elif isinstance(key, Timeframe):
            if key in self._data:
                return DataTimeframe(self, key)
            else:
                raise KeyError(f"No data available for timeframe {key}")
        else:
            raise TypeError(
                f"Invalid key type. Expected int or Timeframe, got {type(key).__name__}"
            )

    @property
    def max_length(self) -> int:
        """Get the maximum number of data points stored for each timeframe."""
        return self._max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        """
        Set the maximum number of data points to store for each timeframe.

        Args:
            value (int): The new maximum length.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if value <= 0:
            logger_main.log_and_raise(
                ValueError("max_length must be a positive integer.")
            )

        old_max_length = self._max_length
        self._max_length = value

        # Update all existing data arrays
        for timeframe in self._data.keys():
            for key in self._data[timeframe].keys():
                self._data[timeframe][key] = self._resize_array(
                    self._data[timeframe][key], value
                )
            self._timestamps[timeframe] = self._resize_array(
                self._timestamps[timeframe], value
            )

        logger_main.info(f"max_length updated from {old_max_length} to {value}")

    @staticmethod
    def _resize_array(arr: np.ndarray, new_length: int) -> np.ndarray:
        """
        Resize a numpy array to a new length, either by padding or trimming.

        Args:
            arr (np.ndarray): The array to resize.
            new_length (int): The desired new length of the array.

        Returns:
            np.ndarray: The resized array.
        """
        if new_length > len(arr):
            pad_width = new_length - len(arr)
            return np.pad(arr, (0, pad_width), "constant", constant_values=np.nan)
        else:
            return arr[:new_length]

    @property
    def timeframes(self) -> List[Timeframe]:
        """Get a list of all available timeframes."""
        return list(self._data.keys())

    @property
    def primary_timeframe(self) -> Timeframe:
        """Get the least available timeframe."""
        return min(self.timeframes)

    def _get_primary_data(self, attr: str) -> np.ndarray:
        """
        Get data for a specific attribute from the primary timeframe.

        Args:
            attr (str): The attribute to get ('open', 'high', 'low', 'close', or 'volume').

        Returns:
            np.ndarray: The requested data as a numpy array.
        """
        return self[self.primary_timeframe][attr]

    @property
    def open(self) -> np.ndarray:
        """Get the open price data for the primary timeframe."""
        return self._get_primary_data("open")

    @property
    def high(self) -> np.ndarray:
        """Get the high price data for the primary timeframe."""
        return self._get_primary_data("high")

    @property
    def low(self) -> np.ndarray:
        """Get the low price data for the primary timeframe."""
        return self._get_primary_data("low")

    @property
    def close(self) -> np.ndarray:
        """Get the close price data for the primary timeframe."""
        return self._get_primary_data("close")

    @property
    def volume(self) -> np.ndarray:
        """Get the volume data for the primary timeframe."""
        return self._get_primary_data("volume")


class DataTimeframe:
    """
    A class to provide convenient access to market data for a specific timeframe.

    This class acts as a view into the Data object, providing easy access to OHLCV data
    for a particular timeframe.

    Attributes:
        _data (Data): The parent Data object this view is associated with.
        _timeframe (Timeframe): The specific timeframe this view represents.
    """

    VALID_KEYS = {"open", "high", "low", "close", "volume"}

    def __init__(self, data: Data, timeframe: Timeframe):
        """
        Initialize the DataTimeframe object.

        Args:
            data (Data): The parent Data object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        self._data = data
        self._timeframe = timeframe

    def __getitem__(self, key: Union[str, int]) -> Union[np.ndarray, Optional[Bar]]:
        """
        Access OHLCV data using dictionary-style key access or get a Bar object by index.

        Args:
            key (Union[str, int]): The data to access ('open', 'high', 'low', 'close', 'volume') or the index of the Bar to retrieve.

        Returns:
            Union[np.ndarray, Optional[Bar]]: The requested data as a numpy array or a Bar object.

        Raises:
            KeyError: If an invalid string key is provided.
            TypeError: If the key is neither a string nor an integer.
        """
        if isinstance(key, str):
            if key not in self.VALID_KEYS:
                raise KeyError(
                    f"Invalid key: {key}. Must be one of {', '.join(self.VALID_KEYS)}."
                )
            return self._data._data[self._timeframe][key]
        elif isinstance(key, int):
            return self._data.get_bar(self._timeframe, key)
        else:
            raise TypeError("Key must be a string or an integer.")

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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the OHLCV data for this timeframe to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data and timestamps.
        """
        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "timestamp": self._data._timestamps[self._timeframe],
            }
        )
