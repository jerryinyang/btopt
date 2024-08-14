from typing import Any, Union

from ..data.timeframe import Timeframe
from ..types import DataManagerType


class ReadOnlyColumnAccessor:
    """
    A class to provide convenient access to a specific column of data.

    This class allows for easy getting of column values using index notation.

    Attributes:
        _data (Data): The parent Data object.
        _timeframe (Timeframe): The timeframe this column is associated with.
        _name (str): The name of the column.
    """

    def __init__(self, data: DataManagerType, timeframe: Timeframe, name: str):
        self._data = data
        self._timeframe = timeframe
        self._name = name

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """
        Get the value(s) of the column at the specified index or slice.

        Args:
            index (Union[int, slice]): The index or slice of the value(s) to retrieve.

        Returns:
            Any: The value(s) of the column at the specified index or slice.
        """
        return self._data._data[self._timeframe][self._name][index]

    def __len__(self) -> int:
        """
        Get the length of the column.

        Returns:
            int: The number of values in the column.
        """
        return len(self._data._data[self._timeframe][self._name])

    def __iter__(self):
        """
        Iterate over the values in the column.

        Yields:
            Any: The next value in the column.
        """
        yield from self._data._data[self._timeframe][self._name]

    def __repr__(self) -> str:
        """
        Return a string representation of the ColumnAccessor object.

        Returns:
            str: A string representation of the object.
        """
        return f"ColumnAccessor(name='{self._name}', timeframe={self._timeframe}, length={len(self)})"


class WritableColumnAccessor(ReadOnlyColumnAccessor):
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
            self._data.set_current(self._timeframe, self._name, value)
        else:
            raise ValueError("Can only set the current (index 0) value of a column")

    def __repr__(self) -> str:
        return f"WritableColumnAccessor(name='{self._name}', timeframe={self._timeframe}, length={len(self)})"
