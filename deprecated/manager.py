from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..util.log_config import logger_main
from .accessor import ReadOnlyColumnAccessor
from .timeframe import Timeframe


class DataManager:
    """
    A base class to manage market data for a specific symbol across multiple timeframes.

    This class provides efficient storage and access to data for different timeframes,
    while maintaining a flexible structure for various data types.

    Attributes:
        symbol (str): The market symbol (e.g., 'EURUSD', 'AAPL') this data represents.
        _max_length (int): The maximum number of data points to store for each timeframe.
        _data (Dict[Timeframe, Dict[str, np.ndarray]]): The internal data storage structure.
        _timestamps (Dict[Timeframe, np.ndarray]): Timestamps for each timeframe.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the DataManager object.

        Args:
            symbol (str): The market symbol this data represents.
            max_length (int, optional): The maximum number of data points to store. Defaults to 500.
        """
        self.symbol: str = symbol
        self._max_length: int = max_length
        self._data: Dict[Timeframe, Dict[str, np.ndarray]] = defaultdict(dict)
        self._timestamps: Dict[Timeframe, np.ndarray] = defaultdict(
            lambda: np.array([], dtype="datetime64[ns]")
        )

    def add_data(
        self, timeframe: Timeframe, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Add new data to the appropriate timeframe.

        Args:
            timeframe (Timeframe): The timeframe of the data.
            timestamp (datetime): The timestamp of the data.
            data (Dict[str, Any]): A dictionary containing the data to add.
        """
        for column, value in data.items():
            if column not in self._data[timeframe]:
                self._data[timeframe][column] = np.empty(self._max_length)
                self._data[timeframe][column].fill(np.nan)

            # Shift the existing data to make room for the new value
            self._data[timeframe][column] = np.roll(self._data[timeframe][column], 1)
            # Add the new value at the beginning
            self._data[timeframe][column][0] = value

        if len(self._timestamps[timeframe]) == 0:
            # Initialize the timestamps array if it's empty
            self._timestamps[timeframe] = np.empty(
                self._max_length, dtype="datetime64[ns]"
            )
            self._timestamps[timeframe].fill(np.datetime64("NaT"))
        else:
            # Shift existing timestamps
            self._timestamps[timeframe] = np.roll(self._timestamps[timeframe], 1)

        # Add the new timestamp at the beginning
        self._timestamps[timeframe][0] = np.datetime64(timestamp)

    def get(
        self,
        timeframe: Optional[Timeframe] = None,
        column: Optional[str] = None,
        index: int = 0,
        size: int = 1,
    ) -> Union[Any, List[Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get data for a given timeframe and column.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe.
            column (Optional[str], optional): The specific column to retrieve. If None, returns all columns.
            index (int, optional): The starting index of the data to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of data points to retrieve. Defaults to 1.

        Returns:
            Union[Any, List[Any], Dict[str, Any], List[Dict[str, Any]]]:
                - If column is specified and size is 1: returns a single value.
                - If column is specified and size > 1: returns a list of values.
                - If column is None and size is 1: returns a dictionary of all columns for a single data point.
                - If column is None and size > 1: returns a list of dictionaries of all columns for multiple data points.

        Raises:
            ValueError: If the specified timeframe or column does not exist.
        """
        if timeframe is None:
            timeframe = self.primary_timeframe

        if timeframe not in self._data:
            raise ValueError(f"No data available for timeframe: {timeframe}")

        if column is not None and column not in self._data[timeframe]:
            raise ValueError(
                f"Column '{column}' does not exist for timeframe: {timeframe}"
            )

        end_index = min(index + size, self._max_length)

        if column is not None:
            data = self._data[timeframe][column][index:end_index]
            return data[0] if size == 1 else data.tolist()
        else:
            result = []
            for i in range(index, end_index):
                point = {
                    col: self._data[timeframe][col][i] for col in self._data[timeframe]
                }
                point["timestamp"] = self._timestamps[timeframe][i]
                result.append(point)
            return result[0] if size == 1 else result

    def __getitem__(self, timeframe: Timeframe) -> "DataTimeframeManager":
        """
        Access data for a specific timeframe.

        Args:
            timeframe (Timeframe): The specific timeframe to access data for.

        Returns:
            DataTimeframeManager: A DataTimeframeManager object providing access to the data for the specified timeframe.

        Raises:
            KeyError: If the specified timeframe does not exist.
        """
        if timeframe not in self._data:
            raise KeyError(f"No data available for timeframe: {timeframe}")
        return DataTimeframeManager(self, timeframe)

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
            raise ValueError("max_length must be a positive integer.")

        old_max_length = self._max_length
        self._max_length = value

        for timeframe in self._data.keys():
            for column in self._data[timeframe].keys():
                self._data[timeframe][column] = self._resize_array(
                    self._data[timeframe][column], value
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
            return np.pad(
                arr, (0, new_length - len(arr)), mode="constant", constant_values=np.nan
            )
        else:
            return arr[:new_length]

    @property
    def timeframes(self) -> List[Timeframe]:
        """Get a list of all available timeframes."""
        return list(self._data.keys())

    @property
    def primary_timeframe(self) -> Timeframe:
        """Get the least available timeframe."""
        return min(self.timeframes) if self.timeframes else None


class DataTimeframeManager:
    """
    A class to provide convenient access to market data for a specific timeframe.

    This class acts as a view into the DataManager object, providing easy access to data
    for a particular timeframe.

    Attributes:
        _data (DataManager): The parent DataManager object this view is associated with.
        _timeframe (Timeframe): The specific timeframe this view represents.
    """

    def __init__(self, data: "DataManager", timeframe: Timeframe):
        """
        Initialize the DataTimeframeManager object.

        Args:
            data (DataManager): The parent DataManager object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        self._data: "DataManager" = data
        self._timeframe: Timeframe = timeframe

    def __getitem__(
        self, key: Union[str, int, slice]
    ) -> Union[np.ndarray, Any, List[Any]]:
        """
        Access data for a specific column or index using dictionary-style key access.

        Args:
            key (Union[str, int, slice]):
                - If str: The name of the column to access.
                - If int or slice: The index or range of data to access.

        Returns:
            Union[np.ndarray, Any, List[Any]]:
                - If key is str: The entire column data.
                - If key is int: A single data point.
                - If key is slice: A list of data points.

        Raises:
            KeyError: If the specified column does not exist.
        """
        if isinstance(key, str):
            if key not in self._data._data[self._timeframe]:
                raise KeyError(
                    f"Column '{key}' does not exist for timeframe: {self._timeframe}"
                )
            return self._data._data[self._timeframe][key]
        elif isinstance(key, (int, slice)):
            return self._data.get(self._timeframe, index=key)
        else:
            raise TypeError("Key must be a string, integer, or slice.")

    def __getattr__(self, name: str) -> ReadOnlyColumnAccessor:
        """
        Access data for a specific column using dot notation.

        Args:
            name (str): The name of the column to access.

        Returns:
            ColumnAccessor: An accessor object for the specified column.

        Raises:
            AttributeError: If the specified column does not exist.
        """
        if name in self._data._data[self._timeframe]:
            return ReadOnlyColumnAccessor(self._data, self._timeframe, name)
        raise AttributeError(
            f"Column '{name}' does not exist for timeframe: {self._timeframe}"
        )

    def __len__(self) -> int:
        """
        Get the number of data points available for this timeframe.

        Returns:
            int: The number of data points.
        """
        return len(next(iter(self._data._data[self._timeframe].values())))

    def __repr__(self) -> str:
        """
        Return a string representation of the DataTimeframeManager object.

        Returns:
            str: A string representation of the object.
        """
        return f"DataTimeframeManager(symbol={self._data.symbol}, timeframe={self._timeframe}, length={len(self)})"
