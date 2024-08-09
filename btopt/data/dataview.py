from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from ..log_config import logger_main
from .timeframe import Timeframe, TimeframeUnit


class DataView:
    """
    A class for managing and aligning financial data across multiple symbols and timeframes.

    This class provides functionality to add, update, align, and retrieve financial data
    for various symbols and timeframes. It maintains a master timeline based on the lowest
    timeframe data and ensures all data is aligned to this timeline.

    Key features:
    - Add and update data for different symbols and timeframes
    - Align all data to a master timeline
    - Resample data to higher timeframes
    - Retrieve specific data points and check if they are original or filled
    - Iterate over aligned data chronologically

    The class is designed to be used in financial backtesting systems, providing a consistent
    and aligned view of data across different symbols and timeframes.
    """

    def __init__(self):
        """
        Initialize the DataView instance.

        This constructor sets up the basic structure for storing and managing financial data.
        It initializes empty data structures and flags that will be populated and updated
        as data is added and processed.

        Attributes:
            data (Dict[str, Dict[Timeframe, pd.DataFrame]]): A nested dictionary to store
                data for each symbol and timeframe.
            master_timeline (pd.DatetimeIndex): The master timeline for all data, based on
                the lowest timeframe. Initially set to None.
            lowest_timeframe (Timeframe): The lowest timeframe among all added data.
                Initially set to None.
            is_aligned (bool): Flag indicating whether all data is aligned to the master
                timeline. Initially set to False.
        """
        self.data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
        self.master_timeline: pd.DatetimeIndex = None
        self.lowest_timeframe: Timeframe = None
        self.is_aligned: bool = False

    def add_data(
        self,
        symbol: str,
        timeframe: Union[str, Timeframe],
        df: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add or update data for a symbol and timeframe without immediate alignment.

        This method adds new data or updates existing data for a given symbol and timeframe.
        If data already exists for the specified symbol and timeframe, the behavior depends
        on the 'overwrite' parameter.

        Args:
            symbol (str): The symbol for which data is being added or updated.
            timeframe (Union[str, Timeframe]): The timeframe of the data. Can be a string
                (e.g., '1m', '1h', '1d') or a Timeframe object.
            df (pd.DataFrame): The dataframe containing the data. Must have a DatetimeIndex
                and include 'open', 'high', 'low', 'close', and 'volume' columns.
            overwrite (bool, optional): Determines how to handle existing data.
                If True, completely replaces any existing data for the symbol/timeframe.
                If False (default), merges new data with existing data, keeping the most
                recent values in case of overlaps.

        Raises:
            ValueError: If the dataframe is empty or doesn't have a DatetimeIndex.
            ValueError: If the dataframe is missing required columns.

        Notes:
            - This method does not immediately align the new data with the master timeline.
            Call 'align_all_data()' after adding all data to perform alignment.
            - Adding or updating data sets the 'is_aligned' flag to False.
            - The lowest timeframe is updated as necessary.

        Example:
            >>> data_manager = DataView()
            >>> df = pd.DataFrame(...)  # Your data here
            >>> data_manager.add_data('AAPL', '1d', df)
            >>> data_manager.add_data('AAPL', '1d', new_df, overwrite=True)  # Replace existing data
        """
        # Ensure the dataframe is not empty
        if df.empty:
            logger_main.log_and_raise(
                ValueError(f"Empty dataframe provided for symbol {symbol}")
            )

        # Ensure the dataframe has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger_main.log_and_raise(
                ValueError(f"Dataframe for symbol {symbol} must have a DatetimeIndex")
            )

        # Convert timeframe to Timeframe object if it's a string
        if isinstance(timeframe, str):
            timeframe = Timeframe(timeframe)

        # Initialize nested dictionary if this is the first data for this symbol
        if symbol not in self.data:
            self.data[symbol] = {}

        # Handle existing data
        if timeframe in self.data[symbol]:
            if overwrite:
                logger_main.warning(
                    f"Overwriting existing data for {symbol} at {timeframe} timeframe.",
                )
                self.data[symbol][timeframe] = df
            else:
                logger_main.info(
                    f"Merging new data with existing data for {symbol} at {timeframe} timeframe.",
                )
                existing_df = self.data[symbol][timeframe]
                merged_df = pd.concat([existing_df, df])
                merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
                merged_df.sort_index(inplace=True)
                self.data[symbol][timeframe] = merged_df
        else:
            # If no existing data, simply add the new data
            self.data[symbol][timeframe] = df

        # Update lowest timeframe
        if self.lowest_timeframe is None or timeframe < self.lowest_timeframe:
            self.lowest_timeframe = timeframe

        # Set the alignment flag to False as new data has been added
        self.is_aligned = False

        logger_main.info(
            f"Data {'added' if timeframe not in self.data[symbol] else 'updated'} for {symbol} at {timeframe} timeframe.",
        )

    def align_all_data(self) -> None:
        """
        Align all data to the master timeline.
        This method should be called after all data has been added and before starting the backtest.
        It calculates the master timeline based on the lowest timeframe data, aligns all data to this timeline,
        and trims higher timeframe data to match the master timeline.
        """
        if self.is_aligned:
            logger_main.info("Data is already aligned.")
            return

        self._calculate_master_timeline()

        if self.master_timeline is None:
            logger_main.info(
                "No master timeline available. Cannot align data.", level="warning"
            )
            return

        for symbol in self.data:
            for timeframe in self.data[symbol]:
                self._align_symbol_data(symbol, timeframe)

        self._trim_higher_timeframe_data()
        self.is_aligned = True
        logger_main.info("All data aligned to master timeline.")

    def _calculate_master_timeline(self):
        """
        Calculate the master timeline based on the lowest timeframe data across all symbols.
        """
        if self.lowest_timeframe is None:
            return

        lowest_tf_data = []
        for symbol_data in self.data.values():
            if self.lowest_timeframe in symbol_data:
                lowest_tf_data.append(symbol_data[self.lowest_timeframe].index)

        if lowest_tf_data:
            self.master_timeline = pd.DatetimeIndex(
                sorted(set().union(*lowest_tf_data))
            )
        else:
            self.master_timeline = None

    def _trim_higher_timeframe_data(self):
        """
        Trim higher timeframe data to match the master timeline.
        """
        for symbol in self.data:
            for timeframe in list(self.data[symbol].keys()):
                if timeframe > self.lowest_timeframe:
                    df = self.data[symbol][timeframe]
                    trimmed_df = df[df.index.isin(self.master_timeline)]
                    self.data[symbol][timeframe] = trimmed_df

    def _align_symbol_data(self, symbol: str, timeframe: Timeframe) -> None:
        """
        Align the data for a specific symbol and timeframe to the master timeline.

        Args:
            symbol (str): The symbol to align.
            timeframe (Timeframe): The timeframe to align.
        """
        df = self.data[symbol][
            timeframe
        ].copy()  # Create a copy to avoid SettingWithCopyWarning
        df = df[~df.index.duplicated(keep="first")]  # Remove duplicates

        # Add a boolean column to indicate original data points
        df.loc[:, "is_original"] = True

        # Reindex the dataframe to the master timeline
        aligned_df = df.reindex(self.master_timeline)

        # Fill missing values and propagate the 'is_original' indicator
        aligned_df = self._custom_fill(aligned_df)

        # Update the stored data with the aligned dataframe
        self.data[symbol][timeframe] = aligned_df

    def _custom_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Custom fill method that forward fills data columns and properly sets the 'is_original' indicator.

        Args:
            df (pd.DataFrame): The dataframe to fill.

        Returns:
            pd.DataFrame: The filled dataframe.
        """
        # Separate 'is_original' column from data columns
        data_columns = [col for col in df.columns if col != "is_original"]

        # Forward fill the data columns
        filled_df = df[data_columns].ffill()

        # Handle 'is_original' column
        if "is_original" in df.columns:
            is_original = df["is_original"]
            filled_df["is_original"] = is_original.astype(float).fillna(False)
        else:
            # If 'is_original' doesn't exist, create it based on non-null values in data columns
            filled_df["is_original"] = df[data_columns].notna().any(axis=1)

        # Ensure 'is_original' is boolean type
        filled_df["is_original"] = filled_df["is_original"].astype(bool)

        return filled_df

    def get_data_point(
        self, symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp
    ) -> pd.Series:
        """
        Get a data point for a specific symbol, timeframe, and timestamp.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.
            timestamp (pd.Timestamp): The timestamp to retrieve data for.

        Returns:
            pd.Series: The data point, or None if no data is available.
        """
        df = self.data[symbol][timeframe]
        if timestamp in df.index:
            return df.loc[timestamp]
        return None

    def is_original_data_point(
        self, symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp
    ) -> bool:
        """
        Check if a data point is an original (non-filled) data point.

        Args:
            symbol (str): The symbol to check.
            timeframe (Timeframe): The timeframe to check.
            timestamp (pd.Timestamp): The timestamp to check.

        Returns:
            bool: True if the data point is original, False otherwise.
        """
        data_point = self.get_data_point(symbol, timeframe, timestamp)

        if data_point is not None:
            return data_point["is_original"]

        return False

    def resample_data(
        self,
        symbol: str,
        from_timeframe: Union[str, Timeframe],
        to_timeframe: Union[str, Timeframe],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Resample data from one timeframe to another (higher) timeframe.

        Args:
            symbol (str): The symbol for which data is being resampled.
            from_timeframe (Union[str, Timeframe]): The original timeframe of the data.
            to_timeframe (Union[str, Timeframe]): The target timeframe for resampling.
            df (pd.DataFrame): The dataframe containing the data to be resampled.

        Returns:
            pd.DataFrame: The resampled dataframe.

        Raises:
            ValueError: If to_timeframe is lower than from_timeframe, or if the dataframe is invalid.
        """
        # Convert timeframes to Timeframe objects if they're strings
        if isinstance(from_timeframe, str):
            from_timeframe = Timeframe(from_timeframe)
        if isinstance(to_timeframe, str):
            to_timeframe = Timeframe(to_timeframe)

        # Check if to_timeframe is higher than from_timeframe
        if to_timeframe <= from_timeframe:
            logger_main.log_and_raise(
                ValueError(
                    f"Target timeframe {to_timeframe} must be higher than original timeframe {from_timeframe}"
                )
            )

        # Validate the input dataframe
        self._validate_dataframe(df, symbol, from_timeframe)

        # Resample the data
        resampled_df = self._resample_dataframe(df, to_timeframe)

        logger_main.info(
            f"Data resampled for {symbol} from {from_timeframe} to {to_timeframe} timeframe.",
        )

        return resampled_df

    def _validate_dataframe(
        self, df: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> None:
        """
        Validate the input dataframe.

        Args:
            df (pd.DataFrame): The dataframe to validate.
            symbol (str): The symbol associated with the dataframe.
            timeframe (Timeframe): The timeframe of the data.

        Raises:
            ValueError: If the dataframe is invalid.
        """
        if df.empty:
            logger_main.log_and_raise(
                ValueError(f"Empty dataframe provided for symbol {symbol}")
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            logger_main.log_and_raise(
                ValueError(f"Dataframe for symbol {symbol} must have a DatetimeIndex")
            )

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger_main.log_and_raise(
                ValueError(
                    f"Dataframe for symbol {symbol} is missing required columns: {missing_columns}"
                )
            )

    def _resample_dataframe(
        self, df: pd.DataFrame, to_timeframe: Timeframe
    ) -> pd.DataFrame:
        # Define the resampling rules
        resample_rule = self._get_resample_rule(to_timeframe)

        # Create a new index with the desired frequency
        new_index = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=resample_rule
        )

        # Reindex the dataframe to the new frequency, forward filling missing values
        resampled = df.reindex(new_index, method="ffill")

        # Aggregate the data
        resampled = resampled.groupby(level=0).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Remove any rows that don't correspond to original data points
        resampled = resampled.loc[resampled.index.isin(df.index)]

        # Add the is_original column
        resampled["is_original"] = True

        return resampled

    def _get_resample_rule(self, timeframe: Timeframe) -> str:
        """
        Get the pandas resample rule for a given timeframe.

        Args:
            timeframe (Timeframe): The timeframe to get the rule for.

        Returns:
            str: The pandas resample rule.
        """
        if timeframe.unit == TimeframeUnit.MINUTE:
            return f"{timeframe.multiplier}min"
        elif timeframe.unit == TimeframeUnit.HOUR:
            return f"{timeframe.multiplier}h"
        elif timeframe.unit == TimeframeUnit.DAY:
            return f"{timeframe.multiplier}d"
        elif timeframe.unit == TimeframeUnit.WEEK:
            return f"{timeframe.multiplier}w"
        elif timeframe.unit == TimeframeUnit.MONTH:
            return f"{timeframe.multiplier}M"
        else:
            logger_main.log_and_raise(
                ValueError(f"Unsupported timeframe unit: {timeframe.unit}")
            )

    def get_data_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the full range of data available in the master timeline.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: A tuple containing the start and end timestamps of the data range.

        Raises:
            ValueError: If no data has been added yet (master timeline is None).
        """
        if self.master_timeline is None:
            logger_main.log_and_raise(
                ValueError(
                    "No data has been added yet. Master timeline is not initialized."
                )
            )

        start_time = self.master_timeline[0]
        end_time = self.master_timeline[-1]

        logger_main.info(f"Data range: from {start_time} to {end_time}")

        return start_time, end_time

    def get_master_timeline(self) -> pd.DatetimeIndex:
        """
        Get the complete master timeline.

        Returns:
            pd.DatetimeIndex: The master timeline containing all timestamps.

        Raises:
            ValueError: If no data has been added yet (master timeline is None).
        """
        if self.master_timeline is None:
            logger_main.log_and_raise(
                ValueError(
                    "No data has been added yet. Master timeline is not initialized."
                )
            )

        return self.master_timeline

    def __iter__(
        self,
    ) -> Iterator[Tuple[pd.Timestamp, Dict[str, Dict[Timeframe, pd.Series]]]]:
        """
        Iterate over the data chronologically.

        Yields:
            Tuple[pd.Timestamp, Dict[str, Dict[Timeframe, pd.Series]]]: A tuple containing the timestamp
            and a dictionary of data points for all symbols and timeframes at that timestamp.
        """
        if self.master_timeline is None or not self.is_aligned:
            logger_main.log_and_raise(
                ValueError(
                    "Data is not ready for iteration. Ensure data is added and aligned."
                )
            )

        for timestamp in self.master_timeline:
            data_point = {}
            for symbol in self.data:
                data_point[symbol] = {}
                for timeframe in self.data[symbol]:
                    if timestamp in self.data[symbol][timeframe].index:
                        data_point[symbol][timeframe] = self.data[symbol][
                            timeframe
                        ].loc[timestamp]
            yield timestamp, data_point

    def get_data(
        self, symbol: str, timeframe: Timeframe, n_bars: int = 1
    ) -> pd.DataFrame:
        """
        Get recent market data for a specific symbol and timeframe.

        Args:
            symbol (str): The symbol to get data for.
            timeframe (Timeframe): The timeframe of the data.
            n_bars (int): The number of recent bars to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the requested market data.
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            logger_main.warning(
                f"No data available for symbol {symbol} and timeframe {timeframe}",
            )
            return pd.DataFrame()

        df = self.data[symbol][timeframe]
        return df.iloc[-n_bars:]

    @property
    def has_data(self) -> bool:
        """
        Check if the DataView contains any data.

        Returns:
            bool: True if data has been added, False otherwise.
        """
        return bool(self.data)

    @property
    def symbols(self) -> List[str]:
        """
        Get the list of symbols in the DataView.

        Returns:
            List[str]: A list of all symbols for which data has been added.
        """
        return list(self.data.keys())

    @property
    def timeframes(self) -> List[Timeframe]:
        """
        Get the list of timeframes in the DataView.

        Returns:
            List[Timeframe]: A list of all unique timeframes across all symbols.
        """
        return list(
            set().union(*[symbol_data.keys() for symbol_data in self.data.values()])
        )
