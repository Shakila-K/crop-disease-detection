"""
Logger — Structured logging setup for the disease detection system.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "govi_disease",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """
    Configure structured logging.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also configure root logger for library modules
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.WARNING)
        root.addHandler(console_handler)

    return logger
