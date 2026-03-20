import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_LEVEL   = os.getenv("DEVA_LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("DEVA_LOG_TO_FILE", "false").lower() == "true"
LOG_FILE    = os.getenv("DEVA_LOG_FILE", "./logs/deva.log")
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"



def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if LOG_TO_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
