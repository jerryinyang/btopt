from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

from ..log_config import logger_main
from ..util.decimal import ExtendedDecimal
from .timeframe import Timeframe


@dataclass
class Bar:
    """
    Represents a single price bar in financial market data.

    This class encapsulates OHLCV (Open, High, Low, Close, Volume) data along with
    timestamp, timeframe, and ticker information. It provides methods for comparison,
    arithmetic operations, and data conversion.

    Attributes:
        open (ExtendedDecimal): The opening price of the bar.
        high (ExtendedDecimal): The highest price of the bar.
        low (ExtendedDecimal): The lowest price of the bar.
        close (ExtendedDecimal): The closing price of the bar.
        volume (int): The trading volume during the bar period.
        timestamp (datetime): The timestamp of the bar.
        timeframe (Timeframe): The timeframe of the bar.
        ticker (str): The ticker symbol of the financial instrument.
        index (Optional[int]): An optional index for the bar, defaults to None.
    """

    open: ExtendedDecimal
    high: ExtendedDecimal
    low: ExtendedDecimal
    close: ExtendedDecimal
    volume: int
    timestamp: datetime
    timeframe: Timeframe
    ticker: str
    index: Optional[int] = None

    def __post_init__(self):
        """
        Perform post-initialization processing.

        Ensures all price fields are of type ExtendedDecimal and validates the bar data.
        """
        # Ensure all price fields are ExtendedDecimal
        for field in ["open", "high", "low", "close"]:
            setattr(self, field, ExtendedDecimal(str(getattr(self, field))))

        # Validate bar data
        self._validate()

    def __getitem__(
        self, key: Union[str, int]
    ) -> Union[ExtendedDecimal, int, datetime, Timeframe, str]:
        """
        Access Bar attributes using dictionary-style key access or list-style index access.

        Args:
            key (Union[str, int]): The key or index to access the attribute.
                - If str: 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'timeframe', 'ticker', 'index'
                - If int: 0 (open), 1 (high), 2 (low), 3 (close), 4 (volume)

        Returns:
            Union[ExtendedDecimal, int, datetime, Timeframe, str]: The value of the requested attribute.

        Raises:
            KeyError: If the string key is not a valid attribute.
            IndexError: If the integer index is out of range.
            TypeError: If the key is neither a string nor an integer.
        """
        if isinstance(key, str):
            if key in self.__annotations__:
                return getattr(self, key)
            else:
                logger_main.error(f"Invalid key: {key}")
                raise KeyError(f"'{key}' is not a valid attribute of Bar")
        elif isinstance(key, int):
            if 0 <= key <= 4:
                return [self.open, self.high, self.low, self.close, self.volume][key]
            else:
                logger_main.error(f"Index out of range: {key}")
                raise IndexError("Bar index out of range")
        else:
            logger_main.error(f"Invalid key type: {type(key)}")
            raise TypeError("Bar indices must be integers or strings")

    # region Validation

    def _validate(self) -> None:
        """
        Validate the bar data to ensure it's logically consistent.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        if self.low > self.high:
            logger_main.error(
                f"Invalid bar data: Low ({self.low}) is greater than High ({self.high})"
            )
            raise ValueError(
                f"Low ({self.low}) cannot be greater than High ({self.high})"
            )

        if self.open < self.low or self.open > self.high:
            logger_main.error(
                f"Invalid bar data: Open ({self.open}) is outside the range of Low ({self.low}) and High ({self.high})"
            )
            raise ValueError(
                f"Open ({self.open}) must be between Low ({self.low}) and High ({self.high})"
            )

        if self.close < self.low or self.close > self.high:
            logger_main.error(
                f"Invalid bar data: Close ({self.close}) is outside the range of Low ({self.low}) and High ({self.high})"
            )
            raise ValueError(
                f"Close ({self.close}) must be between Low ({self.low}) and High ({self.high})"
            )

        if self.volume < 0:
            logger_main.error(f"Invalid bar data: Volume ({self.volume}) is negative")
            raise ValueError(f"Volume ({self.volume}) cannot be negative")

    # endregion

    # region Comparison Methods

    def __eq__(self, other: Any) -> bool:
        """
        Check if this Bar is equal to another Bar.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the Bars are equal, False otherwise.
        """
        if not isinstance(other, Bar):
            return NotImplemented
        return (
            self.timestamp == other.timestamp
            and self.timeframe == other.timeframe
            and self.ticker == other.ticker
        )

    def __lt__(self, other: "Bar") -> bool:
        """
        Check if this Bar is less than another Bar based on their timestamps.

        Args:
            other (Bar): The Bar to compare with.

        Returns:
            bool: True if this Bar's timestamp is earlier than the other Bar's timestamp.

        Raises:
            TypeError: If other is not a Bar object.
        """
        if not isinstance(other, Bar):
            logger_main.error(f"Cannot compare Bar with {type(other)}")
            raise TypeError(
                f"'<' not supported between instances of 'Bar' and '{type(other).__name__}'"
            )
        return self.timestamp < other.timestamp

    def __gt__(self, other: "Bar") -> bool:
        """
        Check if this Bar is greater than another Bar based on their timestamps.

        Args:
            other (Bar): The Bar to compare with.

        Returns:
            bool: True if this Bar's timestamp is later than the other Bar's timestamp.

        Raises:
            TypeError: If other is not a Bar object.
        """
        if not isinstance(other, Bar):
            logger_main.error(f"Cannot compare Bar with {type(other)}")
            raise TypeError(
                f"'>' not supported between instances of 'Bar' and '{type(other).__name__}'"
            )
        return self.timestamp > other.timestamp

    # endregion

    # region Utility Methods

    def fills_price(self, price: ExtendedDecimal) -> bool:
        """
        Check if the given price is within the high and low values of the bar.

        Args:
            price (ExtendedDecimal): The price to check.

        Returns:
            bool: True if the price is within the bar's range, False otherwise.
        """
        return self.low <= price <= self.high

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Bar object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the Bar.
        """
        return {
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": str(self.timeframe),
            "ticker": self.ticker,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bar":
        """
        Create a Bar object from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing Bar data.

        Returns:
            Bar: A new Bar object created from the dictionary data.

        Raises:
            ValueError: If the dictionary is missing required fields or contains invalid data.
        """
        try:
            return cls(
                open=ExtendedDecimal(str(data["open"])),
                high=ExtendedDecimal(str(data["high"])),
                low=ExtendedDecimal(str(data["low"])),
                close=ExtendedDecimal(str(data["close"])),
                volume=(data["volume"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                timeframe=Timeframe(data["timeframe"]),
                ticker=data["ticker"],
                index=data.get("index"),
            )
        except (KeyError, ValueError) as e:
            logger_main.error(f"Error creating Bar from dictionary: {e}")
            raise ValueError(f"Invalid dictionary data: {e}")

    def __repr__(self) -> str:
        """
        Return a string representation of the Bar.

        Returns:
            str: A string representation of the Bar.
        """
        return (
            f"Bar(ticker={self.ticker}, timestamp={self.timestamp}, "
            f"timeframe={self.timeframe}, open={self.open}, high={self.high}, "
            f"low={self.low}, close={self.close}, volume={self.volume})"
        )

    # endregion
