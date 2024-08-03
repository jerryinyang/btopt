from pathlib import Path

from .util.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent

# Define your default logging configuration
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs/main.log"
DEFAULT_LOG_LEVEL = "info"
DEFAULT_CONSOLE_LEVEL = "warning"


# Create a pre-configured logger
logger = get_logger(
    name="main",
    file_location=DEFAULT_LOG_FILE,
    level=DEFAULT_LOG_LEVEL,
    console_level=DEFAULT_CONSOLE_LEVEL,
)
