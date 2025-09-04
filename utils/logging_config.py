import logging
import logging.handlers
import os
from pathlib import Path

project_logger=None

def setup_logging(log_dir: str = "logs", log_file: str = "app.log", level=logging.DEBUG):
    """Setup global logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("Valgenagent_logger")  # root project logger
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger  # already configured

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler (rotating logs)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    project_logger=logger

def get_logger():
     return logging.getLogger("Valgenagent_logger")