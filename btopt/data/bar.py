from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from .timeframe import Timeframe


@dataclass
class Bar:
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
        # Ensure all price fields are Decimal
        self.open = Decimal(str(self.open))
        self.high = Decimal(str(self.high))
        self.low = Decimal(str(self.low))
        self.close = Decimal(str(self.close))

    def fills_price(self, price: Decimal) -> bool:
        """Check if the given price is within the high and low values of the bar."""
        return self.low <= price <= self.high

    def __repr__(self):
        return (
            f"Bar(ticker={self.ticker}, timestamp={self.timestamp}, "
            f"timeframe={self.timeframe}, open={self.open}, high={self.high}, "
            f"low={self.low}, close={self.close}, volume={self.volume})"
        )
