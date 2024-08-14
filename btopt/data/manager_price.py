from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd

from ..util.ext_decimal import ExtendedDecimal
from .bar import Bar
from .manager import DataManager, DataTimeframeManager
from .timeframe import Timeframe


class PriceDataManager(DataManager):
    """
    A class to manage OHLCV (Open, High, Low, Close, Volume) market data for a specific symbol across multiple timeframes.

    This class inherits from the base DataManager class and adds OHLCV-specific functionality.

    Attributes:
        Inherits all attributes from the DataManager class.
    """

    def __init__(self, symbol: str, max_length: int = 500):
        """
        Initialize the PriceDataManager object.

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
            "open": ExtendedDecimal(bar.open),
            "high": ExtendedDecimal(bar.high),
            "low": ExtendedDecimal(bar.low),
            "close": ExtendedDecimal(bar.close),
            "volume": int(bar.volume),
        }
        super().add_data(bar.timeframe, bar.timestamp, data)

    def get(
        self,
        timeframe: Optional[Timeframe] = None,
        column: Optional[str] = None,
        index: int = 0,
        size: int = 1,
    ) -> Union[Any, List[Any], Optional[Bar], List[Optional[Bar]]]:
        """
        Get Bar object(s) or specific column data for a given timeframe and index.

        Args:
            timeframe (Optional[Timeframe], optional): The timeframe to get data for. If None, uses the primary timeframe.
            column (Optional[str], optional): The specific column to retrieve. If None, returns Bar object(s).
            index (int, optional): The starting index of the bar to retrieve (0 is the most recent). Defaults to 0.
            size (int, optional): The number of bars to retrieve. Defaults to 1.

        Returns:
            Union[Any, List[Any], Optional[Bar], List[Optional[Bar]]]:
                - If column is specified: returns the result from the parent get() method.
                - If column is None and size is 1: returns a single Bar object or None if not available.
                - If column is None and size > 1: returns a list of Bar objects (may contain None for missing data).
        """
        # Use the parent get() method to retrieve the data
        data = super().get(timeframe, column, index, size)

        # If a specific column was requested, return the data as-is
        if column is not None:
            return data

        # If no specific column was requested, convert the data to Bar object(s)
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


class PriceDataTimeframe(DataTimeframeManager):
    """
    A class to provide convenient access to OHLCV market data for a specific timeframe.

    This class extends the DataTimeframe class with OHLCV-specific functionality.

    Attributes:
        Inherits all attributes from the DataTimeframe class.
    """

    def __init__(self, data: PriceDataManager, timeframe: Timeframe):
        """
        Initialize the PriceDataTimeframe object.

        Args:
            data (PriceDataManager): The parent PriceDataManager object.
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
