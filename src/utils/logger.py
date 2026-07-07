"""
Logging utilities
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def _resolve_level(level: int | str | None) -> int:
    """Resolve a logging level from an explicit value or the MOODNOTE_LOG_LEVEL env var."""
    if level is None:
        level = os.getenv("MOODNOTE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return level


def setup_logger(
    name: str = "moodnote",
    log_dir: str = "logs",
    log_file: str | None = None,
    level: int | str | None = None,
    log_to_file: bool | None = None,
) -> logging.Logger:
    """
    Setup logger with console and (optionally) file handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name (default: timestamp-based)
        level: Logging level (int or name); defaults to MOODNOTE_LOG_LEVEL or INFO
        log_to_file: Whether to write a log file; defaults to MOODNOTE_LOG_TO_FILE != "0"

    Returns:
        logging.Logger: Configured logger
    """
    level = _resolve_level(level)
    if log_to_file is None:
        log_to_file = os.getenv("MOODNOTE_LOG_TO_FILE", "1") != "0"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate log lines on re-init
    logger.handlers.clear()

    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.debug(f"Logger initialized. Log file: {log_path / log_file}")

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
