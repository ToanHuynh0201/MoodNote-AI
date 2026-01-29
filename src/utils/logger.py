"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "moodnote",
    log_dir: str = "logs",
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name (default: timestamp-based)
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path / log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {log_path / log_file}")

    return logger


def get_logger(name: str = "moodnote") -> logging.Logger:
    """
    Get existing logger or create new one

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger = setup_logger(name)

    return logger


if __name__ == "__main__":
    # Test logger
    logger = setup_logger("test_logger")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
