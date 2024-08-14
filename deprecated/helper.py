from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..util.ext_decimal import ExtendedDecimal
from ..util.log_config import logger_main


class Data:
    """A class to manage market data for a specific symbol across multiple timeframes.

    This class provides efficient storage and access to OHLCV (Open, High, Low, Close, Volume)
    data for different timeframes, while maintaining compatibility with Bar objects.
    It also allows for custom columns to be added and modified, with easy access using dot notation.

    Attributes:
        symbol (str): The market symbol (e.g., 'EURUSD', 'AAPL') this data represents.
        _max_length (int): The maximum number of data points to store for each timeframe.
        _data (Dict[Timeframe, Dict[str, np.ndarray]]): The internal data storage structure.
        _timestamps (Dict[Timeframe, np.ndarray]): Timestamps for each timeframe.
        _custom_columns (Dict[Timeframe, Dict[str, np.ndarray]]): Custom columns for each timeframe.

    Example:
        >>> data = Data(symbol='AAPL', max_length=1000)
        >>> data.add_bar(new_bar)
        >>> latest_close = data[Timeframe('1h')].close[0]
        >>> data.add_custom_column(Timeframe('1h'), 'sma_20')
        >>> data[Timeframe('1h')].sma_20[0] = 150.5  # Set current value
        >>> current_sma = data[Timeframe('1h')].sma_20[0]  # Get current value
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
        self._custom_columns: Dict[Timeframe, Dict[str, np.ndarray]] = defaultdict(dict)

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

        # Update custom columns
        for column, values in self._custom_columns[timeframe].items():
            if len(values) > 0:
                previous_value = values[0]
            else:
                previous_value = np.nan
            self._custom_columns[timeframe][column] = np.concatenate(
                ([previous_value], values[: self.max_length - 1])
            )

    def add_custom_column(self, timeframe: Timeframe, name: str) -> None:
        """
        Add a new custom column to the specified timeframe.

        Args:
            timeframe (Timeframe): The timeframe to add the column to.
            name (str): The name of the new column.
        """
        if name in self._data[timeframe] or name in self._custom_columns[timeframe]:
            logger_main.warning(f"Column '{name}' already exists. Overwriting.")

        self._custom_columns[timeframe][name] = np.full(self._max_length, np.nan)

    def get(
        self,
        timeframe: Optional[Timeframe] = None,
        index: int = 0,
        size: int = 1,
        value: Optional[str] = None,
    ) -> Union[
        Optional[Bar],
        List[Bar],
        Optional[Union[ExtendedDecimal, int, float]],
        List[Union[ExtendedDecimal, int, float]],
    ]:
        """
        Get Bar object(s), specific value(s) for built-in columns, or custom column data for a given timeframe and index.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe. Defaults to None.
            index (int, optional): The starting index of the bar to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of bars to retrieve. Defaults to 1.
            value (Optional[str], optional): If specified, returns only this attribute of the Bar(s) or custom column.
                                            Must be one of 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'timeframe', 'ticker', 'index',
                                            or a custom column name.
                                            Defaults to None (returns full Bar object(s)).

        Returns:
            Union[Optional[Bar], List[Bar], Optional[Union[ExtendedDecimal, int, float]], List[Union[ExtendedDecimal, int, float]]]:
                - If value is None:
                    - If size is 1, returns a single Bar object or None if not available.
                    - If size > 1, returns a list of Bar objects (may be shorter than size if not enough data is available).
                - If value is specified:
                    - If size is 1, returns the specified value (ExtendedDecimal for prices, int for volume, float for custom columns,
                      or appropriate type for other attributes) or None if not available.
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
        valid_values.update(self._custom_columns[timeframe].keys())

        if value and value not in valid_values:
            raise ValueError(
                f"Invalid value: {value}. Must be one of {', '.join(valid_values)}"
            )

        result = []
        for i in range(index, min(index + size, len(self._data[timeframe]["close"]))):
            if value in self._custom_columns[timeframe]:
                result.append(self._custom_columns[timeframe][value][i])
            elif value:
                bar = Bar(
                    open=ExtendedDecimal(str(self._data[timeframe]["open"][i])),
                    high=ExtendedDecimal(str(self._data[timeframe]["high"][i])),
                    low=ExtendedDecimal(str(self._data[timeframe]["low"][i])),
                    close=ExtendedDecimal(str(self._data[timeframe]["close"][i])),
                    volume=int(self._data[timeframe]["volume"][i]),
                    timestamp=self._timestamps[timeframe][i].astype(datetime),
                    timeframe=timeframe,
                    ticker=self.symbol,
                    index=i,
                )
                result.append(getattr(bar, value))
            else:
                result.append(
                    Bar(
                        open=ExtendedDecimal(str(self._data[timeframe]["open"][i])),
                        high=ExtendedDecimal(str(self._data[timeframe]["high"][i])),
                        low=ExtendedDecimal(str(self._data[timeframe]["low"][i])),
                        close=ExtendedDecimal(str(self._data[timeframe]["close"][i])),
                        volume=int(self._data[timeframe]["volume"][i]),
                        timestamp=self._timestamps[timeframe][i].astype(datetime),
                        timeframe=timeframe,
                        ticker=self.symbol,
                        index=i,
                    )
                )

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
                return None
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
    and custom columns for a particular timeframe. It allows access to custom columns
    using dot notation and setting of current values for custom columns.

    Attributes:
        _data (Data): The parent Data object this view is associated with.
        _timeframe (Timeframe): The specific timeframe this view represents.

    Methods:
        __len__: Get the number of data points available for this timeframe.
        __getitem__: Access OHLCV data, custom columns, or Bar objects.
        __setitem__: Set values for a custom column.
        __iter__: Iterate over the Bar objects in this timeframe.
        __repr__: Return a string representation of the DataTimeframe object.
        __getattr__: Access custom columns using dot notation.
        __setattr__: Set values for custom columns using dot notation.
        set_current: Set the current (most recent) value of a column.
        get_custom_column: Get a read-only view of a custom column.
        to_dataframe: Convert the OHLCV data for this timeframe to a pandas DataFrame.
    """

    VALID_KEYS: set = {"open", "high", "low", "close", "volume"}

    def __init__(self, data: Data, timeframe: Timeframe):
        """
        Initialize the DataTimeframe object.

        Args:
            data (Data): The parent Data object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        self._data: Data = data
        self._timeframe: Timeframe = timeframe

    def __len__(self) -> int:
        """
        Get the number of data points available for this timeframe.

        Returns:
            int: The number of data points.
        """
        return len(self._data._data[self._timeframe]["close"])

    def __getitem__(
        self, key: Union[str, int, slice]
    ) -> Union[np.ndarray, Bar, List[Bar], None]:
        """
        Access OHLCV data, custom columns, or Bar objects using dictionary-style key access.

        Args:
            key (Union[str, int, slice]):
                - If str: The data to access ('open', 'high', 'low', 'close', 'volume', or custom column name)
                - If int: The index of the Bar to retrieve
                - If slice: A range of Bars to retrieve

        Returns:
            Union[np.ndarray, Bar, List[Bar], None]:
                - If str: The requested data as a numpy array (read-only for custom columns)
                - If int: A Bar object or None
                - If slice: A list of Bar objects (may contain None for missing data)

        Raises:
            KeyError: If an invalid string key is provided.
            TypeError: If the key is not a string, integer, or slice.
        """
        if isinstance(key, str):
            if key in self.VALID_KEYS:
                return self._data._data[self._timeframe][key]
            elif key in self._data._custom_columns[self._timeframe]:
                # Return a read-only view of the custom column
                return self._data._custom_columns[self._timeframe][key].view()
            else:
                raise KeyError(
                    f"Invalid key: {key}. Must be one of {', '.join(self.VALID_KEYS)} or a custom column name."
                )
        elif isinstance(key, (int, slice)):
            return self._data.get(self._timeframe, index=key)
        else:
            raise TypeError("Key must be a string, integer, or slice.")

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """
        Set values for a custom column.

        Args:
            key (str): The name of the custom column.
            value (np.ndarray): The values to set for the custom column.

        Raises:
            KeyError: If trying to set a value for a non-custom column.
            ValueError: If the length of the value array doesn't match the data length.
        """
        if key in self.VALID_KEYS:
            raise KeyError(f"Cannot modify built-in column '{key}'")

        if key not in self._data._custom_columns[self._timeframe]:
            self._data.add_custom_column(self._timeframe, key)

        if len(value) != len(self._data._data[self._timeframe]["close"]):
            raise ValueError("Value length must match the data length")

        # Create a copy of the input array to ensure immutability
        self._data._custom_columns[self._timeframe][key] = value.copy()

    def __getattr__(self, name: str) -> Union[np.ndarray, "ColumnAccessor"]:
        """
        Access columns using dot notation.

        Args:
            name (str): The name of the column.

        Returns:
            Union[np.ndarray, ColumnAccessor]:
                - If a built-in column: The column data as a numpy array.
                - If a custom column: A ColumnAccessor object for the custom column.

        Raises:
            AttributeError: If the column does not exist.
        """
        if name in self.VALID_KEYS:
            return self._data._data[self._timeframe][name]
        elif name in self._data._custom_columns[self._timeframe]:
            return ColumnAccessor(self._data, self._timeframe, name)
        else:
            raise AttributeError(f"Column '{name}' does not exist")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set values for a custom column using dot notation.

        Args:
            name (str): The name of the custom column.
            value (Any): The values to set for the custom column.

        Raises:
            AttributeError: If trying to set a value for a non-custom column.
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in self._data._custom_columns[self._timeframe]:
            self._data._custom_columns[self._timeframe][name] = value
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}'. Use add_custom_column method first."
            )

    def set_current(self, column_name: str, value: Any) -> None:
        """
        Set the current (most recent) value of a column.

        Args:
            column_name (str): The name of the column.
            value (Any): The value to set for the current position of the column.

        Raises:
            KeyError: If the column_name is not a valid column or doesn't exist.
        """
        if column_name in self.VALID_KEYS:
            raise KeyError(f"Cannot modify built-in column '{column_name}'")

        if column_name not in self._data._custom_columns[self._timeframe]:
            raise KeyError(f"Custom column '{column_name}' does not exist")

        self._data._custom_columns[self._timeframe][column_name][0] = value

    def get_custom_column(self, column_name: str) -> np.ndarray:
        """
        Get a read-only view of a custom column.

        Args:
            column_name (str): The name of the custom column.

        Returns:
            np.ndarray: A read-only view of the custom column.

        Raises:
            KeyError: If the column_name is not a custom column or doesn't exist.
        """
        if column_name in self.VALID_KEYS:
            raise KeyError(f"'{column_name}' is not a custom column")

        if column_name not in self._data._custom_columns[self._timeframe]:
            raise KeyError(f"Custom column '{column_name}' does not exist")

        return self._data._custom_columns[self._timeframe][column_name].view()

    def __iter__(self):
        """
        Iterate over the Bar objects in this timeframe.

        Yields:
            Bar: The next Bar object in the timeframe.
        """
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        """
        Return a string representation of the DataTimeframe object.

        Returns:
            str: A string representation of the object.
        """
        return f"DataTimeframe(symbol={self._data.symbol}, timeframe={self._timeframe}, length={len(self)})"

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

    @property
    def timestamps(self) -> np.ndarray:
        """Get the timestamp data for this timeframe."""
        return self._data._timestamps[self._timeframe]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the OHLCV data for this timeframe to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data and timestamps.
        """
        df = pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "timestamp": self._data._timestamps[self._timeframe],
            }
        )

        # Add custom columns to the DataFrame
        for column_name, column_data in self._data._custom_columns[
            self._timeframe
        ].items():
            df[column_name] = column_data

        return df


class ColumnAccessor:
    """
    A class to provide convenient access to a specific column.

    This class allows for easy getting and setting of column values using index notation.

    Attributes:
        _data (Data): The parent Data object.
        _timeframe (Timeframe): The timeframe this column is associated with.
        _name (str): The name of the column.
    """

    def __init__(self, data: Data, timeframe: Timeframe, name: str):
        """
        Initialize the ColumnAccessor object.

        Args:
            data (Data): The parent Data object.
            timeframe (Timeframe): The timeframe this column is associated with.
            name (str): The name of the column.
        """
        self._data = data
        self._timeframe = timeframe
        self._name = name

    def __getitem__(self, index: int) -> Any:
        """
        Get the value of the column at the specified index.

        Args:
            index (int): The index of the value to retrieve (0 is the most recent).

        Returns:
            Any: The value of the column at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        try:
            return self._data._custom_columns[self._timeframe][self._name][index]
        except IndexError:
            raise IndexError(f"Index {index} is out of range for column '{self._name}'")

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Set the value of the column at the specified index.

        Args:
            index (int): The index of the value to set (0 is the most recent).
            value (Any): The value to set.

        Raises:
            IndexError: If trying to set a value at an index other than 0.
        """
        if index == 0:
            self._data[self._timeframe].set_current(self._name, value)
        else:
            raise IndexError("Can only set the current (index 0) value of a column")

    def __len__(self) -> int:
        """
        Get the length of the column.

        Returns:
            int: The number of values in the column.
        """
        return len(self._data._custom_columns[self._timeframe][self._name])

    def __iter__(self):
        """
        Iterate over the values in the column.

        Yields:
            Any: The next value in the column.
        """
        yield from self._data._custom_columns[self._timeframe][self._name]

    def __repr__(self) -> str:
        """
        Return a string representation of the ColumnAccessor object.

        Returns:
            str: A string representation of the object.
        """
        return f"ColumnAccessor(name='{self._name}', timeframe={self._timeframe}, length={len(self)})"


if __name__ == "__main__":
    # Example usage
    data = Data(symbol="AAPL", max_length=1000)

    # Add a custom column
    data.add_custom_column(Timeframe("1h"), "sma_20")

    # Set and get values using dot notation
    data[Timeframe("1h")].sma_20[0] = 150.5
    current_sma = data[Timeframe("1h")].sma_20[0]

    # Access built-in columns
    latest_close = data[Timeframe("1h")].close[0]

    # Convert to DataFrame
    df = data[Timeframe("1h")].to_dataframe()

    print(f"Current SMA: {current_sma}")
    print(f"Latest close: {latest_close}")
    print(df.head())
