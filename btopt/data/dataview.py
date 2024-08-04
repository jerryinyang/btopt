from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit  # noqa

from ..log_config import logger_main
from .timeframe import Timeframe, TimeframeUnit

pd.set_option("future.no_silent_downcasting", True)


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
                logger_main.log_and_print(
                    f"Overwriting existing data for {symbol} at {timeframe} timeframe.",
                    level="warning",
                )
                self.data[symbol][timeframe] = df
            else:
                logger_main.log_and_print(
                    f"Merging new data with existing data for {symbol} at {timeframe} timeframe.",
                    level="info",
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

        logger_main.log_and_print(
            f"Data {'added' if timeframe not in self.data[symbol] else 'updated'} for {symbol} at {timeframe} timeframe.",
            level="info",
        )

    def align_all_data(self) -> None:
        """
        Align all data to the master timeline.
        This method should be called after all data has been added and before starting the backtest.
        It calculates the master timeline based on the lowest timeframe data, aligns all data to this timeline,
        and trims higher timeframe data to match the master timeline.
        """
        if self.is_aligned:
            logger_main.log_and_print("Data is already aligned.", level="info")
            return

        self._calculate_master_timeline()

        if self.master_timeline is None:
            logger_main.log_and_print(
                "No master timeline available. Cannot align data.", level="warning"
            )
            return

        for symbol in self.data:
            for timeframe in self.data[symbol]:
                self._align_symbol_data(symbol, timeframe)

        self._trim_higher_timeframe_data()
        self.is_aligned = True
        logger_main.log_and_print("All data aligned to master timeline.", level="info")

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
        Custom fill method that propagates both data and the 'is_original' indicator.

        Args:
            df (pd.DataFrame): The dataframe to fill.

        Returns:
            pd.DataFrame: The filled dataframe.
        """

        # Forward fill the data
        filled_df = df.ffill()

        # Infer objects to handle dtype changes
        filled_df = filled_df.infer_objects(copy=False)

        # Propagate the 'is_original' indicator
        filled_df["is_original"] = filled_df["is_original"].fillna(False)

        # Explicitly set the dtype of 'is_original' to boolean
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
    ) -> None:
        """
        Resample data from one timeframe to another (higher) timeframe and add it to the dataset.

        Args:
            symbol (str): The symbol for which data is being resampled.
            from_timeframe (Union[str, Timeframe]): The original timeframe of the data.
            to_timeframe (Union[str, Timeframe]): The target timeframe for resampling.
            df (pd.DataFrame): The dataframe containing the data to be resampled.

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

        # Add the resampled data using add_data method
        self.add_data(symbol, to_timeframe, resampled_df)

        logger_main.log_and_print(
            f"Data resampled and added for {symbol} from {from_timeframe} to {to_timeframe} timeframe.",
            level="info",
        )

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
        """
        Resample a dataframe to a higher timeframe.

        Args:
            df (pd.DataFrame): The dataframe to resample.
            to_timeframe (Timeframe): The target timeframe for resampling.

        Returns:
            pd.DataFrame: The resampled dataframe.
        """
        # Define the resampling rules
        resample_rule = self._get_resample_rule(to_timeframe)

        # Resample the data
        resampled = df.resample(resample_rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

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
            return f"{timeframe.multiplier}T"
        elif timeframe.unit == TimeframeUnit.HOUR:
            return f"{timeframe.multiplier}H"
        elif timeframe.unit == TimeframeUnit.DAY:
            return f"{timeframe.multiplier}D"
        elif timeframe.unit == TimeframeUnit.WEEK:
            return f"{timeframe.multiplier}W"
        elif timeframe.unit == TimeframeUnit.MONTH:
            return f"{timeframe.multiplier}M"
        else:
            raise ValueError(f"Unsupported timeframe unit: {timeframe.unit}")

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

        logger_main.log_and_print(
            f"Data range: from {start_time} to {end_time}", level="info"
        )

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

    def __iter__(self) -> Iterator[Dict[str, Dict[str, pd.Series]]]:
        """
        Iterate over the data chronologically.

        Yields:
            Dict[str, Dict[str, pd.Series]]: A dictionary containing data for all symbols and timeframes
            at each timestamp in the master timeline.
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


class OptimizedDataView:
    """
    A class that provides an optimized view of financial data using numpy arrays.

    This class takes a DataView instance and creates a more memory-efficient and
    faster-to-access representation of the data using numpy arrays. It's designed
    for scenarios where performance is critical, such as in backtesting systems.

    Attributes:
        original_data (DataView): The original DataView instance.
        data_array (np.ndarray): A 4D numpy array containing all the data.
        master_timeline (pd.DatetimeIndex): The master timeline for all data.
        symbol_to_index (Dict[str, int]): Mapping of symbols to their indices.
        timeframe_to_index (Dict[Timeframe, int]): Mapping of timeframes to their indices.
        timestamp_to_index (Dict[pd.Timestamp, int]): Mapping of timestamps to their indices.
        feature_to_index (Dict[str, int]): Mapping of features to their indices in the data array.
    """

    def __init__(self, data_instance: DataView):
        """
        Initialize the OptimizedDataView with a DataView instance.

        Args:
            data_instance (DataView): An aligned DataView instance to optimize.

        Raises:
            ValueError: If the provided DataView instance is not aligned.
        """
        if not data_instance.is_aligned:
            raise ValueError("Data must be aligned before creating optimized view.")
        self.original_data = data_instance
        self.data_array = None
        self.master_timeline = None
        self.symbol_to_index: Dict[str, int] = {}
        self.timeframe_to_index: Dict[Timeframe, int] = {}
        self.timestamp_to_index: Dict[pd.Timestamp, int] = {}
        self.feature_to_index: Dict[str, int] = {
            "open": 0,
            "high": 1,
            "low": 2,
            "close": 3,
            "volume": 4,
            "is_original": 5,
        }
        self._build_optimized_view()

    def _build_optimized_view(self):
        """
        Build the optimized numpy array view of the data.

        This method creates the index mappings and fills the numpy array with data.
        """
        self._build_index_mappings()
        n_symbols = len(self.symbol_to_index)
        n_timeframes = len(self.timeframe_to_index)
        n_timestamps = len(self.timestamp_to_index)
        n_features = len(self.feature_to_index)
        self.data_array = np.full(
            (n_symbols, n_timeframes, n_timestamps, n_features), np.nan
        )
        self._fill_data_array()

    def _build_index_mappings(self):
        """
        Build index mappings for symbols, timeframes, and timestamps.

        This method creates dictionaries that map symbols, timeframes, and timestamps
        to their respective indices in the numpy array.
        """
        self.symbol_to_index = {
            symbol: idx for idx, symbol in enumerate(self.original_data.data.keys())
        }
        all_timeframes = set().union(
            *[
                set(symbol_data.keys())
                for symbol_data in self.original_data.data.values()
            ]
        )
        self.timeframe_to_index = {
            tf: idx for idx, tf in enumerate(sorted(all_timeframes))
        }
        self.master_timeline = self.original_data.master_timeline
        self.timestamp_to_index = {
            ts: idx for idx, ts in enumerate(self.master_timeline)
        }

    def _fill_data_array(self):
        """
        Fill the numpy array with data from the original Data instance.

        This method populates the numpy array with data from the original DataView instance.
        """
        for symbol, symbol_data in self.original_data.data.items():
            symbol_idx = self.symbol_to_index[symbol]
            for timeframe, df in symbol_data.items():
                timeframe_idx = self.timeframe_to_index[timeframe]
                for timestamp, row in df.iterrows():
                    timestamp_idx = self.timestamp_to_index[timestamp]
                    self.data_array[symbol_idx, timeframe_idx, timestamp_idx] = [
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["is_original"],
                    ]

    # @njit
    def get_data_point(
        self, symbol_idx: int, timeframe_idx: int, timestamp_idx: int
    ) -> np.ndarray:
        """
        Get a data point for a specific symbol, timeframe, and timestamp using indices.

        Args:
            symbol_idx (int): Index of the symbol.
            timeframe_idx (int): Index of the timeframe.
            timestamp_idx (int): Index of the timestamp.

        Returns:
            np.ndarray: Array containing the data point.
        """
        return self.data_array[symbol_idx, timeframe_idx, timestamp_idx]

    def get_data_point_by_keys(
        self, symbol: str, timeframe: Timeframe, timestamp: pd.Timestamp
    ) -> np.ndarray:
        """
        Get a data point using string/object keys instead of indices.

        Args:
            symbol (str): The symbol to retrieve data for.
            timeframe (Timeframe): The timeframe to retrieve data for.
            timestamp (pd.Timestamp): The timestamp to retrieve data for.

        Returns:
            np.ndarray: Array containing the data point.

        Raises:
            KeyError: If any of the provided keys are invalid.
        """
        try:
            symbol_idx = self.symbol_to_index[symbol]
            timeframe_idx = self.timeframe_to_index[timeframe]
            timestamp_idx = self.timestamp_to_index[timestamp]
        except KeyError as e:
            raise KeyError(f"Invalid key: {e}")
        return self.get_data_point(symbol_idx, timeframe_idx, timestamp_idx)

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
        data_point = self.get_data_point_by_keys(symbol, timeframe, timestamp)
        return bool(data_point[self.feature_to_index["is_original"]])

    def get_data_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the full range of data available in the master timeline.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp]: A tuple containing the start and end timestamps of the data range.
        """
        return self.master_timeline[0], self.master_timeline[-1]

    def get_master_timeline(self) -> pd.DatetimeIndex:
        """
        Get the complete master timeline.

        Returns:
            pd.DatetimeIndex: The master timeline containing all timestamps.
        """
        return self.master_timeline

    def __iter__(self):
        """
        Iterate over the data chronologically.

        Yields:
            Tuple[pd.Timestamp, Dict]: A tuple containing the timestamp and a dictionary of data points
            for all symbols and timeframes at that timestamp.
        """
        for timestamp_idx, timestamp in enumerate(self.master_timeline):
            data_point = {}
            for symbol, symbol_idx in self.symbol_to_index.items():
                data_point[symbol] = {}
                for timeframe, timeframe_idx in self.timeframe_to_index.items():
                    data = self.get_data_point(symbol_idx, timeframe_idx, timestamp_idx)
                    if not np.isnan(data[0]):
                        data_point[symbol][timeframe] = {
                            feature: data[feature_idx]
                            for feature, feature_idx in self.feature_to_index.items()
                        }
            yield timestamp, data_point

    @property
    def symbols(self) -> List[str]:
        """
        Get the list of symbols.

        Returns:
            List[str]: A list of all symbols in the dataset.
        """
        return list(self.symbol_to_index.keys())

    @property
    def timeframes(self) -> List[Timeframe]:
        """
        Get the list of timeframes.

        Returns:
            List[Timeframe]: A list of all timeframes in the dataset.
        """
        return list(self.timeframe_to_index.keys())
