import logging
import sys
from pathlib import Path
from typing import Literal, Union


class LogFormatter(logging.Formatter):
    FORMAT_DEBUG_INFO = "%(asctime)s - %(levelname)s [%(name)s] - %(message)s"
    FORMAT_WARNING_ERROR = (
        "%(asctime)s - %(levelname)s [%(name)s]\n%(message)s - %(pathname)s:%(lineno)d"
    )

    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style._fmt = self.FORMAT_DEBUG_INFO
        else:
            self._style._fmt = self.FORMAT_WARNING_ERROR
        return super().format(record)


class Logger(logging.Logger):
    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(
        self,
        name: str = __name__,
        file_location: Union[str, Path] = "logs.log",
        level: Literal["debug", "info", "warning", "error", "critical"] = "warning",
        console_level: Literal[
            "debug", "info", "warning", "error", "critical"
        ] = "warning",
    ):
        super().__init__(name, self._parse_level(level))

        formatter = LogFormatter()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._parse_level(console_level))
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(file_location)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    @classmethod
    def _parse_level(cls, level: str) -> int:
        return cls.LEVEL_MAP.get(level.lower(), logging.WARNING)

    def log_and_raise(
        self, error: Exception, level: Literal["error", "critical"] = "error"
    ):
        """
        Log the error message and then raise the exception.

        Args:
            error (Exception): The exception to be logged and raised.
            level (Literal["error", "critical"]): The log level to use. Defaults to "error".

        Raises:
            The provided exception after logging it.
        """
        log_method = getattr(self, level)

        # Get the caller's frame information
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Create a custom record with the correct file and line information
        record = self.makeRecord(
            self.name,
            self.LEVEL_MAP[level],
            filename,
            lineno,
            str(error),
            None,
            None,
            func=frame.f_code.co_name,
        )

        # Log the custom record
        log_method(str(error), extra={"custom_record": record})

        raise error

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        """
        Override the internal _log method to use our custom record when available.
        """
        if extra and "custom_record" in extra:
            record = extra["custom_record"]
        else:
            record = self.makeRecord(
                self.name,
                level,
                self.findCaller(stack_info, stacklevel)[0],
                self.findCaller(stack_info, stacklevel)[1],
                msg,
                args,
                exc_info,
                func=self.findCaller(stack_info, stacklevel)[2],
                extra=extra,
                sinfo=self.findCaller(stack_info, stacklevel)[3]
                if stack_info
                else None,
            )

        self.handle(record)

    def log_and_print(
        self, message: str, level: Literal["info", "debug", "warning"] = "info"
    ):
        """
        Log and print the message.

        Args:
            message (str): The message to be logged and printed.
            level (Literal["info", "debug", "warning"]): The log level to use. Defaults to "info".
        """
        log_method = getattr(self, level)

        # Get the caller's frame information
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        # Create a custom record with the correct file and line information
        record = self.makeRecord(
            self.name,
            self.LEVEL_MAP[level],
            filename,
            lineno,
            message,
            None,
            None,
            func=frame.f_code.co_name,
        )

        # Log the custom record
        log_method(message, extra={"custom_record": record})

        print(message)


def get_logger(
    name: str = __name__,
    file_location: Union[str, Path] = "logs.log",
    level: Literal["debug", "info", "warning", "error", "critical"] = "warning",
    console_level: Literal["debug", "info", "warning", "error", "critical"] = "warning",
) -> Logger:
    """Factory function to create and configure a Logger instance."""
    return Logger(name, file_location, level, console_level)


if __name__ == "__main__":
    # Example usage
    logger = get_logger(level="debug", console_level="info")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
