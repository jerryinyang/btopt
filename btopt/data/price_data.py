from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from ..data.bar import Bar
from ..data.timeframe import Timeframe
from ..util.ext_decimal import ExtendedDecimal
from .data import Data, DataTimeframe


class PriceData(Data):
    """
    A class to manage OHLCV (Open, High, Low, Close, Volume) market data for a specific symbol across multiple timeframes.

    This class inherits from the base Data class and adds OHLCV-specific functionality.

    Attributes:
        Inherits all attributes from the Data class.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the PriceData object.

        Args:
            symbol (str): The market symbol this data represents.
            max_length (int, optional): The maximum number of data points to store. Defaults to 500.
        """
        super().__init__(symbol, max_length)
        self._columns = ["open", "high", "low", "close", "volume"]

    def add_bar(self, bar: Bar) -> None:
        """
        Add a new bar of OHLCV data to the appropriate timeframe.

        Args:
            bar (Bar): The new bar of market data to add.
        """
        data = {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        }
        super().add_data(bar.timeframe, bar.timestamp, data)

    def get_bar(
        self, timeframe: Optional[Timeframe] = None, index: int = 0, size: int = 1
    ) -> Union[Optional[Bar], List[Optional[Bar]]]:
        """
        Get Bar object(s) for a given timeframe and index.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe.
            index (int, optional): The starting index of the bar to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of bars to retrieve. Defaults to 1.

        Returns:
            Union[Optional[Bar], List[Optional[Bar]]]:
                - If size is 1, returns a single Bar object or None if not available.
                - If size > 1, returns a list of Bar objects (may contain None for missing data).
        """
        data = super().get(timeframe, None, index, size)
        if not data:
            return None if size == 1 else []

        if not isinstance(data, list):
            data = [data]

        bars = []
        for item in data:
            if all(col in item for col in self._columns):
                bar = Bar(
                    open=ExtendedDecimal(str(item["open"])),
                    high=ExtendedDecimal(str(item["high"])),
                    low=ExtendedDecimal(str(item["low"])),
                    close=ExtendedDecimal(str(item["close"])),
                    volume=int(item["volume"]),
                    timestamp=item["timestamp"].astype(datetime),
                    timeframe=timeframe or self.primary_timeframe,
                    ticker=self.symbol,
                )
                bars.append(bar)
            else:
                bars.append(None)

        return bars[0] if size == 1 else bars

    def __getitem__(self, timeframe: Timeframe) -> "PriceDataTimeframe":
        """
        Access data for a specific timeframe.

        Args:
            timeframe (Timeframe): The specific timeframe to access data for.

        Returns:
            PriceDataTimeframe: A PriceDataTimeframe object providing access to the OHLCV data for the specified timeframe.

        Raises:
            KeyError: If the specified timeframe does not exist.
        """
        if timeframe not in self._data:
            raise KeyError(f"No data available for timeframe: {timeframe}")
        return PriceDataTimeframe(self, timeframe)


class PriceDataTimeframe(DataTimeframe):
    """
    A class to provide convenient access to OHLCV market data for a specific timeframe.

    This class extends the DataTimeframe class with OHLCV-specific functionality.

    Attributes:
        Inherits all attributes from the DataTimeframe class.
    """

    def __init__(self, data: PriceData, timeframe: Timeframe):
        """
        Initialize the PriceDataTimeframe object.

        Args:
            data (PriceData): The parent PriceData object.
            timeframe (Timeframe): The timeframe this object represents.
        """
        super().__init__(data, timeframe)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the OHLCV data for this timeframe to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data and timestamps.
        """
        import pandas as pd

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
        return df

    def __repr__(self) -> str:
        """
        Return a string representation of the PriceDataTimeframe object.

        Returns:
            str: A string representation of the object.
        """
        return f"PriceDataTimeframe(symbol={self._data.symbol}, timeframe={self._timeframe}, length={len(self)})"
