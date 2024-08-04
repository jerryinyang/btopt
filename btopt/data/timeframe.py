from enum import Enum
from typing import Union

from ..log_config import logger_main


class TimeframeUnit(Enum):
    MINUTE = 1
    HOUR = 60
    DAY = 24 * 60
    WEEK = 7 * 24 * 60
    MONTH = 30 * 24 * 60

    @classmethod
    def from_string(cls, value: str) -> "TimeframeUnit":
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
        try:
            cls.from_string(value)
            return True
        except ValueError:
            return False


class Timeframe:
    def __init__(self, value: Union[str, int, TimeframeUnit]):
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

    @staticmethod
    def _parse_string(value: str) -> tuple[int, TimeframeUnit]:
        numeric_part = ""
        unit_part = ""
        for char in value:
            if char.isdigit():
                numeric_part += char
            else:
                unit_part += char

        if not numeric_part:
            numeric_part = "1"

        multiplier = int(numeric_part)
        unit = TimeframeUnit.from_string(unit_part)
        return multiplier, unit

    @property
    def in_minutes(self) -> int:
        return self.multiplier * self.unit.value

    def __eq__(self, other: "Timeframe") -> bool:
        return self.in_minutes == other.in_minutes

    def __lt__(self, other: "Timeframe") -> bool:
        return self.in_minutes < other.in_minutes

    def __gt__(self, other: "Timeframe") -> bool:
        return self.in_minutes > other.in_minutes

    def __str__(self) -> str:
        unit_str = {
            TimeframeUnit.MINUTE: "m",
            TimeframeUnit.HOUR: "h",
            TimeframeUnit.DAY: "d",
            TimeframeUnit.WEEK: "w",
            TimeframeUnit.MONTH: "mo",
        }[self.unit]
        return f"{self.multiplier}{unit_str}"

    def __repr__(self) -> str:
        return f"Timeframe('{self.__str__()}')"


if __name__ == "__main__":
    # Test cases
    print(Timeframe("1mo"))  # Timeframe(1m)
    print(Timeframe(60))  # Timeframe(60m)
    print(Timeframe(TimeframeUnit.HOUR))  # Timeframe(1h)
    print(Timeframe(TimeframeUnit.HOUR))  # Timeframe(1h)
