from datetime import timedelta
from enum import Enum
from typing import Tuple, Union

from ..log_config import logger_main


class TimeframeUnit(Enum):
    """
    Enumeration of timeframe units with their corresponding values in minutes.
    """

    MINUTE = 1
    HOUR = 60
    DAY = 24 * 60
    WEEK = 7 * 24 * 60
    MONTH = 30 * 24 * 60

    @classmethod
    def from_string(cls, value: str) -> "TimeframeUnit":
        """
        Convert a string representation to a TimeframeUnit.

        Args:
            value (str): String representation of the timeframe unit.

        Returns:
            TimeframeUnit: The corresponding TimeframeUnit.

        Raises:
            ValueError: If the input string is not a recognized timeframe unit.
        """
        mapping = {
            "m": cls.MINUTE,
            "min": cls.MINUTE,
            "h": cls.HOUR,
            "d": cls.DAY,
            "w": cls.WEEK,
            "mo": cls.MONTH,
        }
        normalized_value = value.lower()
        if normalized_value not in mapping:
            error = ValueError(f"Unrecognized timeframe unit: {value}")
            logger_main.log_and_raise(error, level="error")
        return mapping[normalized_value]

    @classmethod
    def is_valid_string(cls, value: str) -> bool:
        """
        Check if a string is a valid timeframe unit representation.

        Args:
            value (str): String to check.

        Returns:
            bool: True if the string is a valid timeframe unit, False otherwise.
        """
        try:
            cls.from_string(value)
            return True
        except ValueError:
            return False


class Timeframe:
    """
    Represents a timeframe with a multiplier and a unit.

    Attributes:
        multiplier (int): The numeric multiplier of the timeframe.
        unit (TimeframeUnit): The unit of the timeframe.
    """

    def __init__(self, value: Union[str, int, TimeframeUnit]):
        """
        Initialize a Timeframe object.

        Args:
            value (Union[str, int, TimeframeUnit]): The timeframe value. Can be a string (e.g., "5m"),
                                                    an integer (interpreted as minutes), or a TimeframeUnit.

        Raises:
            ValueError: If the input value is invalid.
        """
        if isinstance(value, str):
            self.multiplier, self.unit = self._parse_string(value)
        elif isinstance(value, int):
            self.multiplier = value
            self.unit = TimeframeUnit.MINUTE
        elif isinstance(value, TimeframeUnit):
            self.multiplier = 1
            self.unit = value
        else:
            error = ValueError(f"Invalid timeframe value: {value}")
            logger_main.log_and_raise(error, level="error")

    # region Parsing and Conversion Methods

    @staticmethod
    def _parse_string(value: str) -> Tuple[int, TimeframeUnit]:
        """
        Parse a string representation of a timeframe.

        Args:
            value (str): String representation of the timeframe.

        Returns:
            Tuple[int, TimeframeUnit]: The multiplier and unit of the timeframe.

        Raises:
            ValueError: If the string cannot be parsed into a valid timeframe.
        """
        numeric_part = ""
        unit_part = ""
        for char in value:
            if char.isdigit():
                numeric_part += char
            else:
                unit_part += char

        if not numeric_part:
            numeric_part = "1"

        try:
            multiplier = int(numeric_part)
            unit = TimeframeUnit.from_string(unit_part)
            return multiplier, unit
        except ValueError as e:
            logger_main.error(f"Failed to parse timeframe string: {value}")
            raise ValueError(f"Invalid timeframe string: {value}") from e

    @property
    def in_minutes(self) -> int:
        """
        Get the total number of minutes in the timeframe.

        Returns:
            int: The number of minutes.
        """
        return self.multiplier * self.unit.value

    def to_pandas_freq(self) -> str:
        """
        Convert the timeframe to a pandas frequency string.

        Returns:
            str: The pandas frequency string representation of the timeframe.
        """
        unit_map = {
            TimeframeUnit.MINUTE: "T",
            TimeframeUnit.HOUR: "H",
            TimeframeUnit.DAY: "D",
            TimeframeUnit.WEEK: "W",
            TimeframeUnit.MONTH: "M",
        }
        return f"{self.multiplier}{unit_map[self.unit]}"

    def to_timedelta(self) -> timedelta:
        """
        Convert the timeframe to a timedelta object.

        Returns:
            timedelta: A timedelta object representing the duration of the timeframe.
        """
        return timedelta(minutes=self.in_minutes)

    # endregion

    # region Comparison Methods

    def __eq__(self, other: object) -> bool:
        """
        Check if this Timeframe is equal to another Timeframe.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the Timeframes are equal, False otherwise.
        """
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.in_minutes == other.in_minutes

    def __lt__(self, other: "Timeframe") -> bool:
        """
        Check if this Timeframe is less than another Timeframe.

        Args:
            other (Timeframe): The Timeframe to compare with.

        Returns:
            bool: True if this Timeframe is less than the other, False otherwise.
        """
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.in_minutes < other.in_minutes

    def __le__(self, other: "Timeframe") -> bool:
        """
        Check if this Timeframe is less than or equal to another Timeframe.

        Args:
            other (Timeframe): The Timeframe to compare with.

        Returns:
            bool: True if this Timeframe is less than or equal to the other, False otherwise.
        """
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.in_minutes <= other.in_minutes

    def __gt__(self, other: "Timeframe") -> bool:
        """
        Check if this Timeframe is greater than another Timeframe.

        Args:
            other (Timeframe): The Timeframe to compare with.

        Returns:
            bool: True if this Timeframe is greater than the other, False otherwise.
        """
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.in_minutes > other.in_minutes

    def __ge__(self, other: "Timeframe") -> bool:
        """
        Check if this Timeframe is greater than or equal to another Timeframe.

        Args:
            other (Timeframe): The Timeframe to compare with.

        Returns:
            bool: True if this Timeframe is greater than or equal to the other, False otherwise.
        """
        if not isinstance(other, Timeframe):
            return NotImplemented
        return self.in_minutes >= other.in_minutes

    # endregion

    # region Utility Methods

    def __hash__(self) -> int:
        """
        Compute a hash value for the Timeframe.

        Returns:
            int: A hash value for the Timeframe.
        """
        return hash((self.multiplier, self.unit))

    def __str__(self) -> str:
        """
        Get a string representation of the Timeframe.

        Returns:
            str: A string representation of the Timeframe.
        """
        unit_str = {
            TimeframeUnit.MINUTE: "m",
            TimeframeUnit.HOUR: "h",
            TimeframeUnit.DAY: "d",
            TimeframeUnit.WEEK: "w",
            TimeframeUnit.MONTH: "mo",
        }[self.unit]
        return f"{self.multiplier}{unit_str}"

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the Timeframe.

        Returns:
            str: A detailed string representation of the Timeframe.
        """
        return f"Timeframe('{self.__str__()}')"

    # endregion


if __name__ == "__main__":
    # Test cases
    print(Timeframe("1mo"))  # Timeframe('1mo')
    print(Timeframe(60))  # Timeframe('60m')
    print(Timeframe(TimeframeUnit.HOUR))  # Timeframe('1h')
    print(Timeframe("5m") + Timeframe("10m"))  # Timeframe('15m')
    print(Timeframe("1h") * 3)  # Timeframe('3h')
    print(Timeframe("1d").to_pandas_freq())  # 1D
    print(Timeframe("4h").to_timedelta())  # 4:00:00
