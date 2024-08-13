from typing import Any, Union

from ..data.timeframe import Timeframe
from ..strategy.helper import Data


class IndicatorData(Data):
    """
    A class to manage indicator data, extending the functionality of the Data class.

    This class provides easier access to custom columns (indicators) using dot notation
    and allows setting the current value of indicators with simple assignment.

    Attributes:
        Inherits all attributes from the Data class.

    Example:
        >>> indicator_data = IndicatorData(symbol='AAPL', max_length=1000)
        >>> indicator_data.add_custom_column(Timeframe('1h'), 'sma_20')
        >>> indicator_data[Timeframe('1h')].sma_20[0] = 150.5  # Set current value
        >>> current_sma = indicator_data[Timeframe('1h')].sma_20[0]  # Get current value
        >>> historical_sma = indicator_data[Timeframe('1h')].sma_20[5]  # Get historical value
    """

    def __getitem__(self, key: Union[int, Timeframe]) -> "IndicatorDataTimeframe":
        """
        Access data for a specific timeframe.

        Args:
            key (Union[int, Timeframe]):
                - If Timeframe: The specific timeframe to access data for.
                - If int: The index of the timeframe to access (0 for the primary/least timeframe).

        Returns:
            IndicatorDataTimeframe: An IndicatorDataTimeframe object providing access to the data for the specified timeframe.

        Raises:
            IndexError: If the integer index is out of range of available timeframes.
            TypeError: If the key is neither an integer nor a Timeframe.
        """
        if isinstance(key, int):
            timeframes = sorted(self.timeframes)
            if 0 <= key < len(timeframes):
                return IndicatorDataTimeframe(self, timeframes[key])
            else:
                raise IndexError(
                    f"Timeframe index {key} is out of range. Available indices: 0 to {len(timeframes) - 1}"
                )
        elif isinstance(key, Timeframe):
            if key in self._data:
                return IndicatorDataTimeframe(self, key)
            else:
                return None
        else:
            raise TypeError(
                f"Invalid key type. Expected int or Timeframe, got {type(key).__name__}"
            )


class IndicatorDataTimeframe:
    """
    A class to provide convenient access to indicator data for a specific timeframe.

    This class acts as a view into the IndicatorData object, providing easy access to
    custom columns (indicators) for a particular timeframe using dot notation.

    Attributes:
        _data (IndicatorData): The parent IndicatorData object this view is associated with.
        _timeframe (Timeframe): The specific timeframe this view represents.
    """

    def __init__(self, data: IndicatorData, timeframe: Timeframe):
        """
        Initialize the IndicatorDataTimeframe object.

        Args:
            data (IndicatorData): The parent IndicatorData object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        self._data = data
        self._timeframe = timeframe

    def __getattr__(self, name: str) -> "IndicatorColumn":
        """
        Access custom columns using dot notation.

        Args:
            name (str): The name of the custom column (indicator).

        Returns:
            IndicatorColumn: An IndicatorColumn object providing access to the custom column data.

        Raises:
            AttributeError: If the custom column does not exist.
        """
        if name in self._data._custom_columns[self._timeframe]:
            return IndicatorColumn(self._data, self._timeframe, name)
        raise AttributeError(f"Custom column '{name}' does not exist")

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


class IndicatorColumn:
    """
    A class to provide convenient access to a specific indicator column.

    This class allows for easy getting and setting of indicator values using index notation.

    Attributes:
        _data (IndicatorData): The parent IndicatorData object.
        _timeframe (Timeframe): The timeframe this indicator is associated with.
        _name (str): The name of the indicator column.
    """

    def __init__(self, data: IndicatorData, timeframe: Timeframe, name: str):
        """
        Initialize the IndicatorColumn object.

        Args:
            data (IndicatorData): The parent IndicatorData object.
            timeframe (Timeframe): The timeframe this indicator is associated with.
            name (str): The name of the indicator column.
        """
        self._data = data
        self._timeframe = timeframe
        self._name = name

    def __getitem__(self, index: int) -> Any:
        """
        Get the value of the indicator at the specified index.

        Args:
            index (int): The index of the value to retrieve (0 is the most recent).

        Returns:
            Any: The value of the indicator at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        try:
            return self._data._custom_columns[self._timeframe][self._name][index]
        except IndexError:
            raise IndexError(
                f"Index {index} is out of range for indicator '{self._name}'"
            )

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Set the value of the indicator at the specified index.

        Args:
            index (int): The index of the value to set (0 is the most recent).
            value (Any): The value to set.

        Raises:
            IndexError: If trying to set a value at an index other than 0.
        """
        if index == 0:
            self._data[self._timeframe].set_current(self._name, value)
        else:
            raise IndexError("Can only set the current (index 0) value of an indicator")

    def __len__(self) -> int:
        """
        Get the length of the indicator column.

        Returns:
            int: The number of values in the indicator column.
        """
        return len(self._data._custom_columns[self._timeframe][self._name])

    def __iter__(self):
        """
        Iterate over the values in the indicator column.

        Yields:
            Any: The next value in the indicator column.
        """
        yield from self._data._custom_columns[self._timeframe][self._name]

    def __repr__(self) -> str:
        """
        Return a string representation of the IndicatorColumn object.

        Returns:
            str: A string representation of the object.
        """
        return f"IndicatorColumn(name='{self._name}', timeframe={self._timeframe}, length={len(self)})"


if __name__ == "__main__":
    indicator_data = IndicatorData(symbol="AAPL", max_length=1000)
    indicator_data.add_custom_column(Timeframe("1h"), "sma_20")

    # Get the current value of a custom column (indicator)
    current_sma = indicator_data.get(Timeframe("1h"), value="sma_20")

    # Get a historical value of a custom column (indicator)
    historical_sma = indicator_data.get(Timeframe("1h"), index=5, value="sma_20")

    # Get multiple historical values of a custom column (indicator)
    sma_values = indicator_data.get(Timeframe("1h"), index=0, size=10, value="sma_20")

    # Get the current close price
    current_close = indicator_data.get(Timeframe("1h"), value="close")

    # Get a Bar object
    current_bar = indicator_data.get(Timeframe("1h"))

    # Get multiple Bar objects
    bars = indicator_data.get(Timeframe("1h"), index=0, size=5)
