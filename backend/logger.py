import logging
import sys


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    PURPLE = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Formatter that adds color codes based on log level."""

    COLOR_MAP = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.PURPLE,
    }

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, Colors.CYAN)
        message = super().format(record)
        return f"{color}{message}{Colors.END}"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with color output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColorFormatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
