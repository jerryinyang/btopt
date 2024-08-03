from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import mysql.connector as connector
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

from .util.logging import Logger
from .util.functions import clear_terminal, debug  # noqa


class DataLoader(ABC):
    """
    DataLoader is an abstract base class for loading and processing financial data for backtesting.

    Attributes:
        logger (Logger): The logger instance for logging messages.
        DATE_FORMAT (str): The date format used for parsing dates.
        OHLC_COLUMNS (List[str]): The list of column names for OHLC data.
        RESOLUTIONS (List[str]): The list of supported resolutions.
        TIMEFRAMES (Dict[str, bt.TimeFrame]): The mapping of resolutions to backtrader timeframes.
        COMPRESSIONS (Dict[str, int]): The mapping of resolutions to data compressions.

    Args:
        symbol (Union[str, List[str]]): The symbol or list of symbols to load data for.
        resolution (Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d", "1w", "1mo"]): The resolution of the data.
        **period_kwargs: Additional keyword arguments for setting the date range.

    Raises:
        ValueError: If `symbol` or `resolution` is not provided.

    """

    logger = Logger("logger_dataloader")

    # CLASS CONSTANT ATTRIBUTES
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    OHLC_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    RESOLUTIONS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d", "1w", "1M"]
    TIMEFRAMES = {
        "1m": bt.TimeFrame.Minutes,
        "5m": bt.TimeFrame.Minutes,
        "15m": bt.TimeFrame.Minutes,
        "30m": bt.TimeFrame.Minutes,
        "1h": bt.TimeFrame.Minutes,
        "2h": bt.TimeFrame.Minutes,
        "4h": bt.TimeFrame.Minutes,
        "8h": bt.TimeFrame.Minutes,
        "1d": bt.TimeFrame.Days,
        "1w": bt.TimeFrame.Weeks,
        "1M": bt.TimeFrame.Months,
    }
    COMPRESSIONS = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "8h": 480,
        "1d": 1,
        "1w": 1,
        "1M": 1,
    }

    def __init__(
        self,
        symbol: Union[str, List[str]],
        resolution: Literal[
            "1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d", "1w", "1mo"
        ],
        **period_kwargs,
    ) -> None:
        """
        Initialize the DataLoader instance.

        Args:
            symbol (Union[str, List[str]]): The symbol or list of symbols to load data for.
            resolution (Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d", "1w", "1mo"]): The resolution of the data.
            **period_kwargs: Additional keyword arguments for setting the date range.

        Raises:
            ValueError: If `symbol` or `resolution` is not provided.

        """

        # Confirm required arguments passed
        if (symbol is None) or (resolution is None):
            message = "`symbol` and `resolution` are required arguments."
            self.logger.error(message)
            raise ValueError(message)

        # Default Data Feed Arguments
        self.data_args = {
            "open": 0,
            "high": 1,
            "low": 2,
            "close": 3,
            "volume": 4,
            "openinterest": -1,
            "timeframe": None,
            "compression": None,
            "fromdate": datetime.strptime("2022-01-01 00:00:00", self.DATE_FORMAT),
            "todate": datetime.strptime("2022-12-31 00:00:00", self.DATE_FORMAT),
        }

        # Set the symbol, resolution and date range for the object
        self.symbols: List[str] = self._set_symbol(symbol)
        self.resolution = self._set_resolution(resolution)
        self.start_date, self.end_date = self._set_date_range(period_kwargs)

        # OHLC DataFrames
        self._raw_dataframes: Dict[str, pd.DataFrame] = {}
        self.__dataframes: Dict[str, pd.DataFrame] = {}

    @property  # Read Only Getter for self.dataframe
    def dataframes(self):
        """
        Get the dataframes property.

        Returns:
            Dict[str, pd.DataFrame]: The dictionary of dataframes.

        """
        return self.__dataframes

    def load_data(self):
        """
        Run the process of fetching, parsing data, and storing it to self.data_frame.

        Returns:
            Dict[str, pd.DataFrame]: The dictionary of parsed dataframes.

        Raises:
            Exception: If there is an error loading the data.

        """
        # Update the fromdate and todate arguments

        self.data_args = self.data_args.copy()
        self.data_args.update(
            fromdate=datetime.strptime(self.start_date, self.DATE_FORMAT),
            todate=datetime.strptime(self.end_date, self.DATE_FORMAT),
        )

        try:
            # Download (fetch) the data; Assign Raw Data self._raw_dataframes
            self._raw_dataframes = self._fetch_data()

            # Parse Each Downloaded Data
            parsed = {
                ticker: self._parse_data(self._raw_dataframes[ticker])
                for ticker in self._raw_dataframes.keys()
            }

            # Assign Parsed Data to self.dataframes
            self.__dataframes = parsed

            # Clear raw downloaded data
            if self.has_data:
                self._raw_dataframes = None

            return self.__dataframes

        except Exception as e:
            self.logger.error(f"Error Loading Data: {e}")
            raise e

    @abstractmethod
    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch the data for the specified symbols.

        Returns:
            Dict[str, pd.DataFrame]: The dictionary of raw dataframes.

        """
        # Must be over-ridden, to populate the symbol dataframe
        return {}

    def _parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse downloaded data into standard format for backtrader.

        Args:
            data (pd.DataFrame): The downloaded data to parse.

        Returns:
            pd.DataFrame: The parsed data in the desired format.

        """
        columns = ["open", "high", "low", "close", "volume"]
        dataframe = data.copy()
        dataframe.columns = dataframe.columns.str.lower()

        # Make sure the dataframe is not empty, and contains all necessary columns
        if dataframe.empty:
            self.logger.error("Passed Data does not contain any data at all.")
            return None

        if not set(columns).issubset(dataframe.columns):
            print(dataframe.columns)
            self.logger.error("Passed Data does not contain all the necessary columns.")
            return None

        try:
            # Reorder columns into the desired format
            dataframe = dataframe[columns]

            # Estimate volume for forex pairs, with ATR values
            if dataframe["volume"].mode()[0] == 0:
                dataframe["volume"] = ta.atr(
                    dataframe["high"],
                    dataframe["low"],
                    dataframe["close"],
                    1,
                    talib=True,
                )
            dataframe = dataframe[
                (dataframe.index >= self.data_args.get("fromdate"))
                & (dataframe.index <= self.data_args.get("todate"))
            ]
            bt_data = bt.feeds.PandasData(dataname=dataframe, **self.data_args)

        except Exception as e:
            self.logger.error(f"Error parsing data: {e}")
            raise e

        return bt_data

    def _set_symbol(self, symbol: Union[str, List[str]]):
        """
        Set the symbol attribute.

        Args:
            symbol (Union[str, List[str]]): The symbol or list of symbols.

        Returns:
            List[str]: The list of symbols.

        Raises:
            TypeError: If the symbol argument is not a string or list.

        """
        if isinstance(symbol, str):
            return [symbol.upper()]

        elif isinstance(symbol, list):
            return [str_.upper() for str_ in symbol]

        else:
            error_message = f"Unsupported data type for symbol: {type(symbol)}"
            self.logger.error(error_message)
            raise TypeError(error_message)

    def _set_resolution(self, resolution: str) -> str:
        """
        Set the resolution attribute.

        Args:
            resolution (str): The resolution.

        Returns:
            str: The updated resolution.

        """
        resolutions = self.RESOLUTIONS
        default_resolution = "1m"

        # Default resolution to '1d' if the passed resolution is not recognized
        if resolution not in resolutions:
            warning_msg = f'Unsupported resolution: "{resolution}" is not recognized. Defaulting to "{default_resolution}".'
            self.logger.warning(warning_msg)
            resolution = default_resolution

        # Update the data_args
        self.data_args.update(
            timeframe=self.TIMEFRAMES[resolution],
            compression=self.COMPRESSIONS[resolution],
        )

        # Return the resolution
        return resolution

    def _set_date_range(self, period_kwargs: dict = {}) -> tuple:
        """
        Set the data range based on the given period arguments, start date, and end date.

        Args:
            period_kwargs (dict): Dictionary containing period information along with start_date and end_date.

        Returns:
            tuple: A tuple containing formatted start and end dates.

        """
        # Default start date
        _default_start = "2022-01-01"

        # Parse or default start date
        start_date = parse(period_kwargs.get("start_date", _default_start))
        if not isinstance(start_date, datetime):
            start_date = parse(_default_start)

        # Parse or default end date
        end_date = parse(
            period_kwargs.get("end_date", datetime.now().strftime(self.DATE_FORMAT))
        )
        if not isinstance(end_date, datetime):
            end_date = datetime.now()

        # Check for period arguments
        durations = {
            "seconds": period_kwargs.get("seconds", 0),
            "minutes": period_kwargs.get("minutes", 0),
            "hours": period_kwargs.get("hours", 0),
            "days": period_kwargs.get("days", 0),
            "weeks": period_kwargs.get("weeks", 0),
            "months": period_kwargs.get("months", 0),
            "years": period_kwargs.get("years", 0),
        }

        # Update start_date based on period arguments if any
        if (any(value > 0 for value in durations.values())) and (
            start_date == parse(_default_start)
        ):
            duration = {
                _key: _value for (_key, _value) in durations.items() if _value > 0
            }

            start_date = end_date - relativedelta(**duration)

        start_date = start_date.strftime(self.DATE_FORMAT)
        end_date = (end_date + timedelta(days=1)).strftime(self.DATE_FORMAT)

        return min(start_date, end_date), max(start_date, end_date)

    def __getitem__(self, key):
        return self.dataframes[key]

    @property
    def has_data(self):
        """
        Check if the DataLoader has data.

        Returns:
            bool: True if data is available, False otherwise.

        """
        return self.dataframes != {}


class YFDataloader(DataLoader):
    def __init__(
        self,
        symbol: Union[str, List[str]],
        resolution: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
        **period_kwargs,
    ) -> None:
        super().__init__(symbol, resolution, **period_kwargs)

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        try:
            _data_dict = {}

            # Loop Through all symbols in the symbols list
            for symbol in self.symbols:
                data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.resolution,
                )

                data.index = pd.to_datetime(data.index)

                if data.empty:
                    error_message = (
                        "WARNING: Fetch Data Unsuccessful. Object Dataframe did not receive any data."
                        + " Ensure the symbol(s) are valid, and the start/end dates are allowed for that resolution."
                    )
                    self.logger.error(error_message)
                    continue

                _data_dict[symbol] = data

        except Exception as e:
            self.logger.error(f"Fetch Data Unsuccessful: {e}")
            raise e

        # Return dictionary of downloaded datas
        return _data_dict


class CSVDataLoader(DataLoader):
    def __init__(
        self,
        symbol: Union[str, List[str]],
        resolution: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
        **period_kwargs,
    ) -> None:
        super().__init__(symbol, resolution, **period_kwargs)

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        _data_dict = {}

        # Loop Through all symbols in the symbols list
        for symbol in self.symbols:
            try:
                data = self.__get_ticker_data(symbol)
                if data is not None:
                    _data_dict[symbol] = data
                else:
                    raise

            except Exception:
                self.logger.error(f"Error occurred when fetching {symbol} data.")
                continue

        # Return dictionary of downloaded datas
        return _data_dict

    def __get_ticker_data(self, ticker: Optional[str]):
        root = Path(__file__).parents[1]
        path = root / "data"
        _name = f"{ticker}.parquet"

        # Search for the file in all subdirectories
        for _filepath in path.glob(f"**/{_name}"):
            if _filepath.exists():
                data = pd.read_parquet(_filepath)
                data = data[["time", "open", "high", "low", "close", "volume"]]
                data.set_index("time", inplace=True)

                return data

        self.logger.error(f"No file found for {ticker}.")
        return None


class MySQLDataLoader(DataLoader):
    def __init__(
        self,
        symbol: Union[str, List[str]],
        resolution: Literal[
            "1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d", "1w", "1mo"
        ],
        host: str,
        user: str,
        password: str,
        database: str,
        **period_kwargs,
    ) -> None:
        super().__init__(symbol, resolution, **period_kwargs)
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        try:
            _data_dict = {}
            connection = connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            cursor = connection.cursor(dictionary=True)

            # Fetch data for all symbols
            query = f"""
                SELECT * FROM prices
                WHERE tickerid IN ({", ".join([f"'{symbol}'" for symbol in self.symbols])})
                AND time >= '{self.start_date}'
                AND time < '{self.end_date}'
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            data = pd.DataFrame(rows)

            if data.empty:
                self.logger.error("No data fetched from the database.")
                return {}

            data["time"] = pd.to_datetime(data["time"])
            data.set_index("time", inplace=True)

            for symbol in self.symbols:
                symbol_data = data[data["tickerid"] == symbol]
                if symbol_data.empty:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    continue
                symbol_data = symbol_data[self.OHLC_COLUMNS]
                _data_dict[symbol] = symbol_data

            cursor.close()
            connection.close()

            return _data_dict

        except connector.Error as err:
            self.logger.error(f"Error: {err}")
            raise


if __name__ == "__main__":
    #     # aapl = YFDataloader('aapl', '1d', start_date='2019-01-01', end_date='2020-12-31')
    #     # aapl.load_data()

    #     # print(aapl.dataframes)

    #     # pass

    #     btcusdt = CSVDataLoader(['btcusdt', 'ethusdt', 'nvda'], '1h', start_date='2019-01-01', end_date='2020-12-31')
    #     btcusdt.load_data()
    #     print(btcusdt.dataframes)

    mysql_loader = MySQLDataLoader(
        symbol=["BTCUSDT", "GMTUSDT", "EURUSD", "GBPNZD"],
        resolution="1m",
        host="localhost",
        user="root",
        password="password",
        database="ohlc",
        start_date="2022-01-01",
        end_date="2022-12-31",
    )
    data = mysql_loader.load_data()
    print(data)
