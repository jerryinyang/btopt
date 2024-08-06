from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from ..log_config import logger_main
from .timeframe import Timeframe


@dataclass
class Bar:
    """
    Represents a single price bar in financial market data.

    This class encapsulates OHLCV (Open, High, Low, Close, Volume) data along with
    timestamp, timeframe, and ticker information. It provides methods for comparison,
    arithmetic operations, and data conversion.

    Attributes:
        open (Decimal): The opening price of the bar.
        high (Decimal): The highest price of the bar.
        low (Decimal): The lowest price of the bar.
        close (Decimal): The closing price of the bar.
        volume (int): The trading volume during the bar period.
        timestamp (datetime): The timestamp of the bar.
        timeframe (Timeframe): The timeframe of the bar.
        ticker (str): The ticker symbol of the financial instrument.
        index (Optional[int]): An optional index for the bar, defaults to None.
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    timestamp: datetime
    timeframe: Timeframe
    ticker: str
    index: Optional[int] = None

    def __post_init__(self):
        """
        Perform post-initialization processing.

        Ensures all price fields are of type Decimal and validates the bar data.
        """
        # Ensure all price fields are Decimal
        for field in ["open", "high", "low", "close"]:
            setattr(self, field, Decimal(str(getattr(self, field))))

        # Validate bar data
        self._validate()

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

    def fills_price(self, price: Decimal) -> bool:
        """
        Check if the given price is within the high and low values of the bar.

        Args:
            price (Decimal): The price to check.

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
                open=Decimal(str(data["open"])),
                high=Decimal(str(data["high"])),
                low=Decimal(str(data["low"])),
                close=Decimal(str(data["close"])),
                volume=int(data["volume"]),
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
