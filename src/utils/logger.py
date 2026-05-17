"""
Structured logging setup for the project.
"""
import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with structured formatting.

    Args:
        name: Logger name (use __name__ in calling module).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
