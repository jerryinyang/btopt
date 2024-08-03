import logging
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
