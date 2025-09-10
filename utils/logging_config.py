import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def setup_logging(log_level=logging.WARNING):
    """Setup global logging configuration."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"vca_{timestamp}.log"
    Path('logs').mkdir(parents=True, exist_ok=True)
    log_path = os.path.join('logs', log_file)
    
    default_logger = logging.getLogger()
    default_logger.setLevel(logging.WARNING) # default: suppress 3rd party logs (INFO/DEBUG)

    logger = logging.getLogger("VCA") 
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger  # already configured

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_stdout_handler = logging.StreamHandler()
    console_stdout_handler.setLevel(logging.DEBUG)
    console_stdout_handler.setFormatter(log_format)
    logger.addHandler(console_stdout_handler)

    console_stderr_handler = logging.StreamHandler()
    console_stderr_handler.setLevel(logging.WARNING)
    console_stderr_handler.setFormatter(log_format)
    logger.addHandler(console_stderr_handler)

    # File handler (rotating logs)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
