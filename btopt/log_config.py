from pathlib import Path

from .util.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent

# Define your default logging configuration
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs/main.log"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_CONSOLE_LEVEL = "error"


def clear_log_files():
    def clear(log_file_path: Path):
        """
        Clears the content of the specified log file.

        :param log_file_path: Path to the log file to be cleared.
        """

        try:
            # Ensure the directory exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open the file in write mode to clear its content
            with open(log_file_path, "w") as file:
                file.truncate(0)
        except Exception:
            return

    paths = [
        DEFAULT_LOG_FILE,
        PROJECT_ROOT / "logs/test.log",
    ]

    for path in paths:
        clear(path)


# Create a pre-configured logger
logger_main = get_logger(
    name="main",
    file_location=DEFAULT_LOG_FILE,
    level="warning",
    console_level=DEFAULT_CONSOLE_LEVEL,
)

logger_test = get_logger(
    name="test",
    file_location=PROJECT_ROOT / "logs/test.log",
    level="info",
    console_level=DEFAULT_CONSOLE_LEVEL,
)
