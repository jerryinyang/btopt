from pathlib import Path

from .util.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent

# Define your default logging configuration
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs/main.log"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_CONSOLE_LEVEL = "error"


def clear_log_file():
    """
    Clears the content of the specified log file.

    :param log_file_path: Path to the log file to be cleared.
    """
    log_file_path = DEFAULT_LOG_FILE
    try:
        # Ensure the directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file in write mode to clear its content
        with open(log_file_path, "w") as file:
            file.truncate(0)
        print(f"Log file '{log_file_path}' has been cleared successfully.")
    except Exception as e:
        print(f"An error occurred while clearing the log file: {e}")


# Create a pre-configured logger
logger_main = get_logger(
    name="main",
    file_location=DEFAULT_LOG_FILE,
    level=DEFAULT_LOG_LEVEL,
    console_level=DEFAULT_CONSOLE_LEVEL,
)
