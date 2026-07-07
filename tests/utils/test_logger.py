"""Tests for the logging setup helpers."""

import logging

from src.utils.logger import get_logger, setup_logger


def test_get_logger_returns_logger_with_matching_name():
    logger = get_logger("moodnote_test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "moodnote_test_logger"


def test_get_logger_does_not_duplicate_handlers():
    logger1 = get_logger("moodnote_test_dedup")
    handler_count = len(logger1.handlers)
    logger2 = get_logger("moodnote_test_dedup")
    assert len(logger2.handlers) == handler_count


def test_setup_logger_reads_level_from_env(monkeypatch):
    monkeypatch.setenv("MOODNOTE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MOODNOTE_LOG_TO_FILE", "0")
    logger = setup_logger("moodnote_test_env_level")
    assert logger.level == logging.DEBUG
