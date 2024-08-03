import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BAR:
    open: float
    high: float
    low: float
    close: float
    volume: float = field(default=0)

    @property
    def direction(self) -> int:
        if self.open < self.close:
            return 1
        elif self.open > self.close:
            return -1
        else:
            return 0


def pivothigh(window: list, period: int, period_right: Optional[int] = None):
    if period_right is None:
        period_right = period

    minimum_length = period + period_right + 1
    assert (
        len(window) >= minimum_length
    ), f"Window length ({len(window)}) is less than what is required for the given period lengths ({minimum_length})."

    # Get the [minimum_length] most recent data points
    data = window[: minimum_length + 1]

    return np.max(data[:period] + (data[period + 1 :])) < data[period]


def pivotlow(window: list, period: int, period_right: Optional[int] = None):
    if period_right is None:
        period_right = period

    minimum_length = period + period_right + 1
    assert (
        len(window) >= minimum_length
    ), f"Window length ({len(window)}) is less than what is required for the given period lengths ({minimum_length})."

    # Get the [minimum_length] most recent data points
    data = window[: minimum_length + 1]

    return np.min(data[:period] + (data[period + 1 :])) > data[period]


def rising(window: list, period: Optional[int] = None):
    if period is None:
        period = len(window) + 1

    data = window[:period]

    return np.min(np.diff(data)) >= 0


def falling(window: list, period: Optional[int] = None):
    if period is None:
        period = len(window) + 1

    data = window[:period]

    return np.max(np.diff(data)) <= 0


def na(value):
    """
    Check if a value is considered as missing or not available.

    Parameters:
    - value: Any, the value to be checked.

    Returns:
    bool: True if the value is missing or not available, False otherwise.
    """
    if isinstance(value, float) and math.isnan(value):
        return True
    elif isinstance(value, np.ndarray) and np.isnan(value).any():
        return True
    elif value is None:
        return True
    return False


def ternary(condition, value_true, value_false):
    """
    Ternary operator implementation.

    Parameters:
    - condition: bool, the condition to check.
    - value_true: Any, the value to return if the condition is True.
    - value_false: Any, the value to return if the condition is False.

    Returns:
    Any: Either value_true or value_false based on the condition.
    """
    if condition:
        return value_true

    return value_false


def nz(value, replacement=0):
    """
    Replace missing or not available values with a specified replacement.

    Parameters:
    - value: Any, the value to be checked for being missing or not available.
    - replacement: Any, the value to be returned if 'value' is missing or not available. Default is 0.

    Returns:
    Any: Either the original 'value' or the specified 'replacement' based on whether 'value' is missing or not.

    Example:
    >>> nz(42)
    42

    >>> nz(None)
    0

    >>> nz(float('nan'), replacement=99)
    99
    """
    return ternary(na(value), replacement, value)


if __name__ == "__main__":
    print(type((1)))
