import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mysql.connector as connector
import pandas as pd
import yfinance as yf
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from pandas.api.types import is_datetime64_any_dtype

from ..util.log_config import logger_main
from .timeframe import Timeframe

load_dotenv()


class BaseDataLoader(ABC):
    """
    Abstract base class for creating specific dataloader classes.

    This class provides a framework for loading financial data for one or more symbols
    over a specified time period and timeframe. It handles data fetching, parsing,
    and basic processing.

    Attributes:
        DATETIME_FORMAT (str): The standard datetime format used in the class.
        OHLC_COLUMNS (List[str]): The standard column names for OHLC data.
        symbols (List[str]): The list of symbols to fetch data for.
        timeframe (Timeframe): The timeframe of the data.
        start_date (str): The start date for data retrieval.
        end_date (str): The end date for data retrieval.
    """

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    OHLC_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(
        self,
        symbol: Union[str, List[str]],
        timeframe: str,
        **period_kwargs: Union[int, float],
    ) -> None:
        """
        Initialize the BaseDataLoader.

        Args:
            symbol (Union[str, List[str]]): The symbol or list of symbols to fetch data for.
            timeframe (str): The timeframe for the data.
            **period_kwargs: Additional keyword arguments for specifying the time period.
                Can include 'start_date', 'end_date', or duration parameters like 'days', 'weeks', etc.
        """
        self.symbols: List[str] = self._set_symbol(symbol)
        self.timeframe = Timeframe(value=timeframe)
        self.start_date, self.end_date = self._set_date_range(period_kwargs)
        self._raw_dataframes: Dict[str, pd.DataFrame] = {}
        self.__dataframes: Dict[str, pd.DataFrame] = {}

        # Load that data
        self._load_data()

    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get the processed dataframes.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes, keyed by symbol.
        """
        return self.__dataframes

    @property
    def tickers(self):
        return self.dataframes.keys()

    def create_modified(
        self,
        modifier_func: Callable[[Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]],
    ) -> "BaseDataLoader":
        """
        Create a new dataloader instance with modified dataframes.

        Args:
            modifier_func (Callable[[Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]]):
                A function that takes the current dataframes dictionary and returns a modified version.

        Returns:
            BaseDataLoader: A new instance of the dataloader with modified data.

        Raises:
            ValueError: If the dataloader does not contain any data.
        """
        if not self.has_data:
            logger_main.log_and_raise(
                ValueError("Dataloader does not contain any data.")
            )

        new_instance = deepcopy(self)
        new_instance._raw_dataframes = None
        new_instance.__dataframes = modifier_func(self.dataframes)
        return new_instance

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and process data for all symbols.

        This method fetches raw data, processes it, and stores the result in the dataframes property.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of processed dataframes, keyed by symbol.

        Raises:
            RuntimeError: If there's an error during data loading or processing.
        """
        try:
            self._raw_dataframes = self._fetch_data()
            self.__dataframes = {
                ticker: self._parse_data(df)
                for ticker, df in self._raw_dataframes.items()
            }
            self._raw_dataframes = None  # Clear raw data after processing
            return self.__dataframes
        except ValueError as e:
            logger_main.log_and_raise(RuntimeError(f"Error Loading Data: {e}"))
        except IOError as e:
            logger_main.log_and_raise(RuntimeError(f"I/O Error Loading Data: {e}"))

    @staticmethod
    def _set_symbol(symbol: Union[str, List[str]]) -> List[str]:
        """
        Set and validate the symbol(s).

        Args:
            symbol (Union[str, List[str]]): The symbol or list of symbols to validate.

        Returns:
            List[str]: A list of uppercase symbol strings.

        Raises:
            ValueError: If symbol is None.
            TypeError: If symbol is neither a string nor a list.
        """
        if symbol is None:
            logger_main.log_and_raise(
                ValueError("`symbol` is a required argument. Cannot be of NoneType")
            )

        if isinstance(symbol, str):
            return [symbol.upper()]
        elif isinstance(symbol, list):
            return [str(s).upper() for s in symbol]
        else:
            logger_main.log_and_raise(
                TypeError(f"Unsupported data type for symbol: {type(symbol)}")
            )

    def _set_date_range(self, period_kwargs: dict) -> tuple:
        """
        Set the date range for data retrieval.

        Args:
            period_kwargs (dict): Keyword arguments specifying the time period.

        Returns:
            tuple: A tuple containing the start and end dates as strings in the standard format.
        """
        default_start = datetime(2022, 1, 1)
        end_date = parse(
            period_kwargs.get("end_date", datetime.now().strftime(self.DATETIME_FORMAT))
        )

        if period_kwargs.get("start_date"):
            start_date = parse(period_kwargs["start_date"])
        else:
            durations = {
                key: period_kwargs.get(key, 0)
                for key in [
                    "seconds",
                    "minutes",
                    "hours",
                    "days",
                    "weeks",
                    "months",
                    "years",
                ]
            }
            if any(durations.values()):
                start_date = end_date - relativedelta(**durations)
            else:
                start_date = default_start

        return (
            min(start_date, end_date).strftime(self.DATETIME_FORMAT),
            (max(start_date, end_date) + timedelta(days=1)).strftime(
                self.DATETIME_FORMAT
            ),
        )

    @abstractmethod
    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch the data for the specified symbols.

        This method must be implemented by subclasses to define how data is retrieved.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of raw dataframes, keyed by symbol.
        """
        pass

    def _parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and process the fetched data.

        Args:
            data (pd.DataFrame): The raw data to process.

        Returns:
            pd.DataFrame: The processed dataframe with standardized datetime index.

        Raises:
            ValueError: If the fetched data is empty or missing required columns.
        """
        if data.empty:
            logger_main.log_and_raise(ValueError("Fetched data is empty."))

        # Ensure column names are lowercase
        data.columns = data.columns.str.lower()

        # Check for missing columns
        missing_columns = set(self.OHLC_COLUMNS) - set(data.columns)
        if missing_columns:
            logger_main.log_and_raise(
                ValueError(
                    f"Fetched data is missing columns: {missing_columns}. Got these columns: {data.columns}"
                )
            )

        # Ensure the index is a datetime index and standardize its format
        if not is_datetime64_any_dtype(data.index):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger_main.log_and_raise(
                    ValueError(f"Unable to convert index to datetime: {e}")
                )

        # # Standardize the datetime format
        # data.index = data.index.strftime(self.DATETIME_FORMAT)

        # Directly convert to datetime with the specified format
        data.index = pd.to_datetime(data.index, format=self.DATETIME_FORMAT)

        # Select only the required columns
        data = data[self.OHLC_COLUMNS]

        # # Estimate volume for forex pairs, with ATR values if volume is zero
        # if data["volume"].mode()[0] == 0:
        #     data.loc[:, "volume"] = ta.atr(
        #         data["high"], data["low"], data["close"], 1, talib=True
        #     ).astype("float64")

        # Filter data based on the specified date range
        start_date = pd.to_datetime(self.start_date, format=self.DATETIME_FORMAT)
        end_date = pd.to_datetime(self.end_date, format=self.DATETIME_FORMAT)
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        return data

    def __getitem__(self, key: str) -> pd.DataFrame:
        """
        Allow dictionary-like access to dataframes.

        Args:
            key (str): The symbol to retrieve data for.

        Returns:
            pd.DataFrame: The dataframe for the specified symbol.
        """
        return self.dataframes[key]

    @property
    def has_data(self) -> bool:
        """
        Check if the DataLoader has data.

        Returns:
            bool: True if data is available, False otherwise.
        """
        return bool(self.dataframes)


class YFDataloader(BaseDataLoader):
    """
    Yahoo Finance data loader class.

    This class extends BaseDataLoader to fetch financial data from Yahoo Finance.
    """

    VALID_TIMEFRAMES = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "90m": "90m",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1w": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }

    def __init__(
        self, symbol: Union[str, List[str]], timeframe: str, **period_kwargs
    ) -> None:
        """
        Initialize the YFDataloader.

        Args:
            symbol (Union[str, List[str]]): The symbol or list of symbols to fetch data for.
            timeframe (str): The timeframe for the data.
            **period_kwargs: Additional keyword arguments for specifying the time period.
        """
        self.yf_timeframe = self._parse_timeframe(timeframe)
        super().__init__(symbol, timeframe, **period_kwargs)

    def _parse_timeframe(self, timeframe: str) -> str:
        """
        Parse the input timeframe to a format recognized by yfinance.

        Args:
            timeframe (str): The input timeframe.

        Returns:
            str: The parsed timeframe suitable for yfinance.

        Raises:
            ValueError: If the timeframe is not supported by yfinance.
        """
        tf = Timeframe(timeframe)
        parsed_tf = f"{tf.multiplier}{tf.unit.name[0].lower()}"

        if parsed_tf not in self.VALID_TIMEFRAMES:
            logger_main.log_and_raise(
                ValueError(
                    f"Unsupported timeframe: {timeframe}. "
                    f"Supported timeframes are: {', '.join(self.VALID_TIMEFRAMES.keys())}"
                )
            )

        return self.VALID_TIMEFRAMES[parsed_tf]

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols from Yahoo Finance.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes, keyed by symbol.

        Raises:
            RuntimeError: If there's an error during data fetching.
        """
        try:
            results = {}
            with ThreadPoolExecutor() as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_symbol_data, symbol): symbol
                    for symbol in self.symbols
                }
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None:
                            results[symbol] = data
                    except Exception as e:
                        logger_main.log_and_raise(
                            f"Error fetching data for {symbol}: {e}"
                        )
            return results
        except Exception as e:
            logger_main.log_and_raise(f"Error in data fetching process: {e}")
            logger_main.log_and_raise(RuntimeError(f"Failed to fetch data: {e}"))

    def _fetch_symbol_data(self, symbol: str) -> Union[pd.DataFrame, None]:
        """
        Fetch data for a single symbol.

        Args:
            symbol (str): The symbol to fetch data for.

        Returns:
            Union[pd.DataFrame, None]: The fetched data as a DataFrame, or None if fetching failed.
        """
        start_date = (
            datetime.strptime(self.start_date, self.DATETIME_FORMAT).date().isoformat()
        )
        end_date = (
            datetime.strptime(self.end_date, self.DATETIME_FORMAT).date().isoformat()
        )

        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=self.yf_timeframe,
            )

            if data.empty:
                logger_main.warning(
                    f"No data received for {symbol}. Ensure the symbol is valid "
                    f"and the date range ({start_date} to {end_date}) "
                    f"is allowed for the {self.yf_timeframe} timeframe."
                )
                return None

            data.index = pd.to_datetime(data.index)
            data["Close"] = data["Adj Close"]
            return data

        except Exception as e:
            logger_main.log_and_raise(f"Failed to fetch data for {symbol}: {e}")
            return None


class CSVDataLoader(BaseDataLoader):
    def __init__(
        self, symbol: Union[str, List[str]], timeframe: str, **period_kwargs
    ) -> None:
        self.data_dir = self._get_data_directory()
        super().__init__(symbol, timeframe, **period_kwargs)

    @staticmethod
    def _get_data_directory() -> Path:
        data_dir = os.getenv("CSV_DATA_DIRECTORY")
        if not data_dir:
            logger_main.warning(
                "CSV_DATA_DIRECTORY not set in .env file. Using default path."
            )
            return Path(__file__).parents[1] / "data"
        return Path(data_dir)

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol in self.symbols:
            try:
                data = self._get_ticker_data(symbol)
                if data is not None:
                    results[symbol] = data
            except Exception as e:
                logger_main.log_and_raise(f"Error fetching data for {symbol}: {e}")
        return results

    def _get_ticker_data(self, symbol: str) -> Optional[pd.DataFrame]:
        file_name = f"{symbol}.parquet"

        # Search for the file recursively in data_dir and its subfolders
        for file_path in self.data_dir.rglob(file_name):
            if file_path.is_file():
                try:
                    data = pd.read_parquet(file_path)

                    # Check if all required OHLC columns are present
                    if not all(col in data.columns for col in self.OHLC_COLUMNS):
                        logger_main.warning(
                            f"File found for {symbol} at {file_path}, but it doesn't contain all required OHLC columns."
                        )
                        continue

                    # Identify the timestamp column
                    timestamp_columns = [
                        col
                        for col in data.columns
                        if "time" in col.lower() or "date" in col.lower()
                    ]
                    if not timestamp_columns:
                        logger_main.warning(
                            f"File found for {symbol} at {file_path}, but it doesn't contain a recognizable timestamp column."
                        )
                        continue

                    timestamp_column = timestamp_columns[0]

                    # Select and rename columns
                    selected_columns = [timestamp_column] + self.OHLC_COLUMNS
                    data = data[selected_columns]
                    data.rename(columns={timestamp_column: "time"}, inplace=True)

                    # Set the index
                    data.set_index("time", inplace=True)

                    logger_main.info(
                        f"Found and loaded data for {symbol} from {file_path}"
                    )
                    return data
                except Exception as e:
                    logger_main.log_and_raise(
                        f"Error reading parquet file for {symbol} at {file_path}: {e}"
                    )
                    continue

        logger_main.warning(
            f"No valid file found for {symbol} in {self.data_dir} or its subfolders."
        )
        return None


class MySQLDataLoader(BaseDataLoader):
    def __init__(
        self,
        symbol: Union[str, List[str]],
        timeframe: str,
        host: str,
        user: str,
        password: str,
        database: str,
        **period_kwargs,
    ) -> None:
        super().__init__(symbol, timeframe, **period_kwargs)
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        try:
            connection = connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            cursor = connection.cursor(dictionary=True)

            query = f"""
                SELECT * FROM prices
                WHERE tickerid IN ({', '.join(['%s' for _ in self.symbols])})
                AND time >= %s AND time < %s
            """
            cursor.execute(query, self.symbols + [self.start_date, self.end_date])
            rows = cursor.fetchall()
            data = pd.DataFrame(rows)

            cursor.close()
            connection.close()

            if data.empty:
                logger_main.warning("No data fetched from the database.")
                return {}

            data["time"] = pd.to_datetime(data["time"])
            data.set_index("time", inplace=True)

            results = {}
            for symbol in self.symbols:
                symbol_data = data[data["tickerid"] == symbol]
                if not symbol_data.empty:
                    results[symbol] = symbol_data[self.OHLC_COLUMNS]
                else:
                    logger_main.warning(f"No data found for symbol {symbol}")

            return results

        except connector.Error as err:
            logger_main.log_and_raise(f"MySQL Error: {err}")
            logger_main.log_and_raise(
                RuntimeError(f"Failed to fetch data from MySQL: {err}")
            )


if __name__ == "__main__":
    data = CSVDataLoader(["EURUSD", "MGBP"], "1m")

    print(data._load_data())
