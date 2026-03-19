# deva/logger.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_LEVEL = os.getenv("DEVA_LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("DEVA_LOG_TO_FILE", "false").lower() == "true"
LOG_FILE_PATH = os.getenv("DEVA_LOG_FILE", "./logs/deva.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Call this at the top of every module:
        from deva.logger import get_logger
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Always log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optionally log to a rotating file
    if LOG_TO_FILE:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,               # keep last 3 rotated files
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate logs)
    logger.propagate = False

    return logger
