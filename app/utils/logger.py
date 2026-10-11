import logging
import sys
from pathlib import Path

def setup_logger(name: str = "MovieMatch") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        fh = logging.FileHandler("app.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger

logger = setup_logger()
