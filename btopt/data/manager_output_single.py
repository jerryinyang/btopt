from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..util.ext_decimal import ExtendedDecimal
from ..util.log_config import logger_main
from .timeframe import Timeframe


class SingleTimeframeOutputManager:
    """
    A class to manage output data for a specific symbol with a single timeframe.

    This class provides functionality for managing output data columns,
    including a placeholder column.

    Attributes:
        symbol (str): The market symbol this data represents.
        timeframe (Timeframe): The timeframe for this output data.
        _max_length (int): The maximum number of data points to store.
        _data (Dict[str, np.ndarray]): The internal data storage structure.
        _timestamps (np.ndarray): Timestamps for the data.
        _placeholder_column (str): The name of the placeholder column.
    """

    def __init__(self, symbol: str, timeframe: Timeframe, max_length: int = 500):
        """
        Initialize the SingleTimeframeOutputManager object.

        Args:
            symbol (str): The market symbol this data represents.
            timeframe (Timeframe): The timeframe for this output data.
            max_length (int, optional): The maximum number of data points to store. Defaults to 500.
        """
        self.symbol: str = symbol
        self.timeframe: Timeframe = timeframe
        self._max_length: int = max_length
        self._data: Dict[str, np.ndarray] = {}
        self._timestamps: np.ndarray = np.empty(max_length, dtype="datetime64[ns]")
        self._timestamps.fill(np.datetime64("NaT"))
        self._placeholder_column = "_placeholder"
        self._columns = [self._placeholder_column]
        self._data[self._placeholder_column] = np.full(self._max_length, np.nan)

    def add_data(self, timestamp: datetime, data: Dict[str, Any]) -> None:
        """
        Add new output data.

        Args:
            timestamp (datetime): The timestamp of the data.
            data (Dict[str, Any]): A dictionary containing the output data to add.
        """
        # Ensure the placeholder column exists and is updated
        if self._placeholder_column not in data:
            data[self._placeholder_column] = np.nan

        # Add any new columns
        for column in data.keys():
            if column not in self._columns:
                self.add_output_column(column)

        # Shift existing data and add new data
        for column, value in data.items():
            self._data[column] = np.concatenate(([value], self._data[column][:-1]))

        # Shift timestamps and add new timestamp
        self._timestamps = np.concatenate(
            ([np.datetime64(timestamp)], self._timestamps[:-1])
        )

    def update_timestamp(self, timestamp: datetime) -> None:
        """
        Start a new iteration by adding a new timestamp and duplicating the most recent values for all columns.

        This method should be called at the beginning of each new iteration to ensure that all columns
        have a value for the new timestamp, even if they haven't been explicitly set.

        Args:
            timestamp (datetime): The timestamp for the new iteration.
        """
        new_data = {}
        for column in self._data.keys():
            if column != self._placeholder_column:
                new_data[column] = self._data[column][
                    0
                ]  # Duplicate the most recent value

        # Add the placeholder column
        new_data[self._placeholder_column] = np.nan

        self.add_data(timestamp, new_data)
        logger_main.info(f"Started new iteration at timestamp {timestamp}")

    def add_output_column(self, column_name: str) -> None:
        """
        Add a new output column.

        Args:
            column_name (str): The name of the new column.

        Raises:
            ValueError: If the column already exists.
        """
        if column_name in self._data:
            raise ValueError(f"Column '{column_name}' already exists")

        self._data[column_name] = np.empty(self._max_length)
        self._data[column_name].fill(np.nan)
        self._columns.append(column_name)
        logger_main.info(f"Added output column '{column_name}'")

    def get(
        self, column: Optional[str] = None, index: int = 0, size: int = 1
    ) -> Union[
        ExtendedDecimal,
        List[ExtendedDecimal],
        Dict[str, ExtendedDecimal],
        List[Dict[str, ExtendedDecimal]],
    ]:
        """
        Get output data for a given column.

        Args:
            column (Optional[str], optional): The specific column to retrieve. If None, returns all columns except the placeholder.
            index (int, optional): The starting index of the data to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of data points to retrieve. Defaults to 1.

        Returns:
            Union[ExtendedDecimal, List[ExtendedDecimal], Dict[str, ExtendedDecimal], List[Dict[str, ExtendedDecimal]]]:
                - If column is specified and size is 1: returns a single value.
                - If column is specified and size > 1: returns a list of values.
                - If column is None and size is 1: returns a dictionary of all columns (except placeholder) for a single data point.
                - If column is None and size > 1: returns a list of dictionaries of all columns (except placeholder) for multiple data points.

        Raises:
            ValueError: If the specified column does not exist.
        """
        end_index = min(index + size, self._max_length)

        if column is not None:
            if column not in self._data:
                raise ValueError(f"Column '{column}' does not exist")
            data = self._data[column][index:end_index]
            return data[0] if size == 1 else data.tolist()
        else:
            result = []
            for i in range(index, end_index):
                point = {
                    col: self._data[col][i]
                    for col in self._data
                    if col != self._placeholder_column
                }
                point["timestamp"] = self._timestamps[i]
                result.append(point)
            return result[0] if size == 1 else result

    def set_current(self, column: str, value: Any) -> None:
        """
        Set the current (most recent) value of an output column.

        Args:
            column (str): The name of the column to update.
            value (Any): The value to set for the current position of the column.

        Raises:
            ValueError: If the column does not exist.
        """
        if column not in self._data:
            raise ValueError(f"Column '{column}' does not exist")

        self._data[column][0] = value
        logger_main.info(f"Updated current value of column '{column}'")

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
            KeyError: If the specified column does not exist or is the placeholder column.
        """
        if isinstance(key, str):
            if key == self._placeholder_column:
                raise KeyError("Cannot access placeholder column directly")
            return self._data[key]
        elif isinstance(key, (int, slice)):
            return self.get(index=key)
        else:
            raise TypeError("Key must be a string, integer, or slice.")

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        """
        Set the current (most recent) value of an output column.

        Args:
            key (Union[str, int]):
                - If str: The name of the column to set.
                - If int: Must be 0 to set the most recent value.
            value (Any): The value to set.

        Raises:
            KeyError: If the column does not exist or is the placeholder column.
            ValueError: If trying to set a value for an index other than 0.
        """
        if isinstance(key, str):
            if key == self._placeholder_column:
                raise KeyError("Cannot set placeholder column directly")
            self.set_current(key, value)
        elif isinstance(key, int):
            if key != 0:
                raise ValueError("Can only set the current (index 0) value of a column")
            raise ValueError("Column name must be specified when setting a value")
        else:
            raise TypeError("Key must be a string or 0")

    def __getattr__(self, name: str) -> "ColumnAccessor":
        """
        Access data for a specific column using dot notation.

        Args:
            name (str): The name of the column to access.

        Returns:
            ColumnAccessor: An accessor object for the specified column.

        Raises:
            AttributeError: If the specified column does not exist or is the placeholder column.
        """
        if name == self._placeholder_column:
            raise AttributeError("Cannot access placeholder column directly")
        if name in self._data:
            return ColumnAccessor(self, name)
        raise AttributeError(f"Column '{name}' does not exist")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the output data to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the output data and timestamps, excluding the placeholder column.
        """
        data = {
            col: self._data[col]
            for col in self._data
            if col != self._placeholder_column
        }
        data["timestamp"] = self._timestamps
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        """
        Return a string representation of the SingleTimeframeOutputManager object.

        Returns:
            str: A string representation of the object.
        """
        return f"SingleTimeframeOutputManager(symbol={self.symbol}, timeframe={self.timeframe}, columns={len(self._data) - 1})"


class ColumnAccessor:
    """
    A class to provide convenient access to a specific column of data.

    This class allows for easy getting and setting of column values using index notation.

    Attributes:
        _manager (SingleTimeframeOutputManager): The parent SingleTimeframeOutputManager object.
        _name (str): The name of the column.
    """

    def __init__(self, manager: SingleTimeframeOutputManager, name: str):
        self._manager = manager
        self._name = name

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """
        Get the value(s) of the column at the specified index or slice.

        Args:
            index (Union[int, slice]): The index or slice of the value(s) to retrieve.

        Returns:
            Any: The value(s) of the column at the specified index or slice.
        """
        return self._manager._data[self._name][index]

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Set the value of the column at the specified index.

        Args:
            index (int): The index of the value to set (0 is the most recent).
            value (Any): The value to set.

        Raises:
            ValueError: If trying to set a value at an index other than 0.
        """
        if index == 0:
            self._manager.set_current(self._name, value)
        else:
            raise ValueError("Can only set the current (index 0) value of a column")

    def __len__(self) -> int:
        """
        Get the length of the column.

        Returns:
            int: The number of values in the column.
        """
        return len(self._manager._data[self._name])

    def __iter__(self):
        """
        Iterate over the values in the column.

        Yields:
            Any: The next value in the column.
        """
        yield from self._manager._data[self._name]

    def __repr__(self) -> str:
        """
        Return a string representation of the ColumnAccessor object.

        Returns:
            str: A string representation of the object.
        """
        return f"ColumnAccessor(name='{self._name}', length={len(self)})"
