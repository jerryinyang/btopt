import logging
import sys
from pathlib import Path
from typing import Any, Literal, Union


class CustomFormatter(logging.Formatter):
    """
    A custom formatter that applies different formats based on the log level.
    This formatter attempts to create clickable links for both filename and line number.
    """

    def __init__(self):
        super().__init__()
        self.formats = {
            logging.DEBUG: "%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
            logging.INFO: "%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
            logging.WARNING: '%(asctime)s - %(levelname)s [%(name)s]\n%(message)s\n  File "%(pathname)s", line %(lineno)d',
            logging.ERROR: '%(asctime)s - %(levelname)s [%(name)s]\n%(message)s\n  File "%(pathname)s", line %(lineno)d',
            logging.CRITICAL: '%(asctime)s - %(levelname)s [%(name)s]\n%(message)s\n  File "%(pathname)s", line %(lineno)d',
        }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record based on its level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.
        """
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger(logging.Logger):
    """
    A custom Logger class that extends the functionality of the standard logging.Logger.

    This Logger provides additional methods for logging and raising exceptions,
    as well as logging and printing messages. It attempts to create clickable links
    for both filename and line number in error messages.
    """

    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(
        self,
        name: str,
        file_location: Union[str, Path],
        level: Literal["debug", "info", "warning", "error", "critical"],
        console_level: Literal["debug", "info", "warning", "error", "critical"],
    ):
        """
        Initialize the Logger instance.

        Args:
            name (str): The name of the logger.
            file_location (Union[str, Path]): The location of the log file.
            level (Literal["debug", "info", "warning", "error", "critical"]): The logging level for the file handler.
            console_level (Literal["debug", "info", "warning", "error", "critical"]): The logging level for the console handler.
        """
        super().__init__(name, self._parse_level(level))

        self.file_location = Path(file_location)
        self.formatter = CustomFormatter()

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._parse_level(console_level))
        console_handler.setFormatter(self.formatter)
        self.addHandler(console_handler)

        # Set up file handler
        file_handler = logging.FileHandler(self.file_location)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.addHandler(file_handler)

    @classmethod
    def _parse_level(cls, level: str) -> int:
        """
        Parse the string log level to its corresponding integer value.

        Args:
            level (str): The string representation of the log level.

        Returns:
            int: The integer value of the log level.
        """
        return cls.LEVEL_MAP.get(level.lower(), logging.WARNING)

    def log_and_raise(
        self,
        error: Exception,
        level: Literal["error", "critical"] = "error",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Log the error message and then raise the exception.

        This method attempts to create a clickable link for both the filename and line number
        in the logged error message.

        Args:
            error (Exception): The exception to be logged and raised.
            level (Literal["error", "critical"]): The log level to use. Defaults to "error".
            *args: Additional positional arguments for the log message.
            **kwargs: Additional keyword arguments for the log message.

        Raises:
            The provided exception after logging it.
        """
        log_method = getattr(self, level)

        # Get caller information
        caller_frame = sys._getframe(1)
        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno

        # Create a message with potentially clickable filename and line number
        message = f'{str(error)}\n  File "{filename}", line {lineno}'

        log_method(message, exc_info=True, stack_info=True, *args, **kwargs)
        raise error

    def log_and_print(
        self,
        message: str,
        level: Literal["debug", "info", "warning"] = "info",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Log the message and print it to the console if the level is appropriate.

        Args:
            message (str): The message to be logged and potentially printed.
            level (Literal["debug", "info", "warning"]): The log level to use. Defaults to "info".
            *args: Additional positional arguments for the log message.
            **kwargs: Additional keyword arguments for the log message.
        """
        log_method = getattr(self, level)
        log_method(message, *args, **kwargs)

        # Print to console if the level is at or above the console handler's level
        if self.LEVEL_MAP[level] >= self.handlers[0].level:
            print(f"{level.upper()}: {message}")


def get_logger(
    name: str = __name__,
    file_location: Union[str, Path] = "logs.log",
    level: Literal["debug", "info", "warning", "error", "critical"] = "warning",
    console_level: Literal["debug", "info", "warning", "error", "critical"] = "warning",
) -> Logger:
    """
    Create and return a configured Logger instance.

    This function creates a Logger instance with potentially clickable links
    for both filename and line number in error messages.

    Args:
        name (str): The name of the logger. Defaults to the calling module's name.
        file_location (Union[str, Path]): The location of the log file. Defaults to "logs.log".
        level (Literal["debug", "info", "warning", "error", "critical"]): The logging level for the file handler. Defaults to "warning".
        console_level (Literal["debug", "info", "warning", "error", "critical"]): The logging level for the console handler. Defaults to "warning".

    Returns:
        Logger: A configured Logger instance.
    """
    return Logger(name, file_location, level, console_level)


# Example usage
if __name__ == "__main__":
    logger = get_logger(level="debug", console_level="info")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    logger.log_and_print("This message will be logged and printed", level="info")

    try:
        logger.log_and_raise(ValueError("This is a test error"), level="error")
    except ValueError as e:
        print(f"Caught exception: {e}")
