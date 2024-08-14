from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..util.log_config import logger_main
from .accessor import WritableColumnAccessor
from .manager import DataManager, DataTimeframeManager
from .timeframe import Timeframe


class MultiTimframeOuputManager(DataManager):
    """
    A class to manage output data for a specific symbol across multiple timeframes.

    This class inherits from the base Data class and adds functionality for managing
    output data columns, including a placeholder column.

    Attributes:
        Inherits all attributes from the Data class.
        _placeholder_column (str): The name of the placeholder column.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the MultiTimframeOuputManager object.

        Args:
            symbol (str): The market symbol this data represents.
            max_length (int, optional): The maximum number of data points to store. Defaults to 500.
        """
        super().__init__(symbol, max_length)
        self._placeholder_column = "_placeholder"
        self._columns = [self._placeholder_column]

    def add_data(
        self, timeframe: Timeframe, timestamp: datetime, data: Dict[str, Any]
    ) -> None:
        """
        Add new output data to the appropriate timeframe.

        Args:
            timeframe (Timeframe): The timeframe of the data.
            timestamp (datetime): The timestamp of the data.
            data (Dict[str, Any]): A dictionary containing the output data to add.
        """
        # Ensure the placeholder column exists and is updated
        if self._placeholder_column not in data:
            data[self._placeholder_column] = np.nan

        # Add any new columns
        for column in data.keys():
            if column not in self._columns:
                self.add_output_column(timeframe, column)
                self._columns.append(column)

        super().add_data(timeframe, timestamp, data)

    def add_output_column(self, timeframe: Timeframe, column_name: str) -> None:
        """
        Add a new output column to the specified timeframe.

        Args:
            timeframe (Timeframe): The timeframe to add the column to.
            column_name (str): The name of the new column.

        Raises:
            ValueError: If the column already exists.
        """
        if column_name in self._data[timeframe]:
            raise ValueError(
                f"Column '{column_name}' already exists for timeframe {timeframe}"
            )

        self._data[timeframe][column_name] = np.full(self._max_length, np.nan)
        logger_main.info(
            f"Added output column '{column_name}' to timeframe {timeframe}"
        )

    def get(
        self,
        timeframe: Optional[Timeframe] = None,
        column: Optional[str] = None,
        index: int = 0,
        size: int = 1,
    ) -> Union[Any, List[Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get output data for a given timeframe and column.

        This method overrides the base get method to exclude the placeholder column from the results.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe.
            column (Optional[str], optional): The specific column to retrieve. If None, returns all columns except the placeholder.
            index (int, optional): The starting index of the data to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of data points to retrieve. Defaults to 1.

        Returns:
            Union[Any, List[Any], Dict[str, Any], List[Dict[str, Any]]]:
                - If column is specified and size is 1: returns a single value.
                - If column is specified and size > 1: returns a list of values.
                - If column is None and size is 1: returns a dictionary of all columns (except placeholder) for a single data point.
                - If column is None and size > 1: returns a list of dictionaries of all columns (except placeholder) for multiple data points.

        Raises:
            ValueError: If the specified timeframe or column does not exist.
        """
        result = super().get(timeframe, column, index, size)

        if column is not None or result is None:
            return result

        if isinstance(result, dict):
            return {k: v for k, v in result.items() if k != self._placeholder_column}
        elif isinstance(result, list):
            return [
                {k: v for k, v in item.items() if k != self._placeholder_column}
                for item in result
            ]

    def set_current(self, timeframe: Timeframe, column: str, value: Any) -> None:
        """
        Set the current (most recent) value of a output column.

        Args:
            timeframe (Timeframe): The timeframe of the data.
            column (str): The name of the column to update.
            value (Any): The value to set for the current position of the column.

        Raises:
            ValueError: If the column does not exist for the specified timeframe.
        """
        if column not in self._data[timeframe]:
            raise ValueError(
                f"Column '{column}' does not exist for timeframe {timeframe}"
            )

        self._data[timeframe][column][0] = value
        logger_main.info(
            f"Updated current value of column '{column}' for timeframe {timeframe}"
        )

    def __getitem__(self, timeframe: Timeframe) -> "MTFOuputTimeframeManager":
        """
        Access data for a specific timeframe.

        Args:
            timeframe (Timeframe): The specific timeframe to access data for.

        Returns:
            MTFOuputTimeframeManager: A MTFOuputTimeframeManager object providing access to the output data for the specified timeframe.

        Raises:
            KeyError: If the specified timeframe does not exist.
        """
        if timeframe not in self._data:
            raise KeyError(f"No data available for timeframe: {timeframe}")
        return MTFOuputTimeframeManager(self, timeframe)


class MTFOuputTimeframeManager(DataTimeframeManager):
    """
    A class to provide convenient access to output market data for a specific timeframe.

    This class extends the DataTimeframe class with functionality specific to output data.

    Attributes:
        Inherits all attributes from the DataTimeframe class.
    """

    def __init__(self, data: MultiTimframeOuputManager, timeframe: Timeframe):
        """
        Initialize the MTFOuputTimeframeManager object.

        Args:
            data (MultiTimframeOuputManager): The parent MultiTimframeOuputManager object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        super().__init__(data, timeframe)

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
            if key == self._data._placeholder_column:
                raise KeyError("Cannot access placeholder column directly")
            return self._data._data[self._timeframe][key]
        elif isinstance(key, (int, slice)):
            return self._data.get(self._timeframe, index=key)
        else:
            raise TypeError("Key must be a string, integer, or slice.")

    def __setitem__(self, key: Union[str, int], value: Any) -> None:
        """
        Set the current (most recent) value of a output column.

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
            if key == self._data._placeholder_column:
                raise KeyError("Cannot set placeholder column directly")
            self._data.set_current(self._timeframe, key, value)
        elif isinstance(key, int):
            if key != 0:
                raise ValueError("Can only set the current (index 0) value of a column")
            raise ValueError("Column name must be specified when setting a value")
        else:
            raise TypeError("Key must be a string or 0")

    def __getattr__(self, name: str) -> WritableColumnAccessor:
        """
        Access data for a specific column using dot notation.

        Args:
            name (str): The name of the column to access.

        Returns:
            WritableColumnAccessor: An accessor object for the specified column.

        Raises:
            AttributeError: If the specified column does not exist or is the placeholder column.
        """
        if name == self._data._placeholder_column:
            raise AttributeError("Cannot access placeholder column directly")
        if name in self._data._data[self._timeframe]:
            return WritableColumnAccessor(self._data, self._timeframe, name)
        raise AttributeError(
            f"Column '{name}' does not exist for timeframe: {self._timeframe}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the output data for this timeframe to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the output data and timestamps, excluding the placeholder column.
        """
        data = {
            col: self._data._data[self._timeframe][col]
            for col in self._data._data[self._timeframe]
            if col != self._data._placeholder_column
        }
        data["timestamp"] = self._data._timestamps[self._timeframe]

        return pd.DataFrame(data)

    def __repr__(self) -> str:
        """
        Return a string representation of the MTFOuputTimeframeManager object.

        Returns:
            str: A string representation of the object.
        """
        return f"MTFOuputTimeframeManager(symbol={self._data.symbol}, timeframe={self._timeframe}, columns={len(self._data._data[self._timeframe]) - 1})"
