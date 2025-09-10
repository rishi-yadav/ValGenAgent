import logging
import logging.handlers
import os
from pathlib import Path
import sys
from datetime import datetime


def setup_logging(log_level=logging.WARNING, project_namespace=None):
    """Setup global logging configuration."""
    import pdb;pdb.set_trace()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"vca_{timestamp}.log"
    Path('logs').mkdir(parents=True, exist_ok=True)
    log_path = os.path.join('logs', log_file)
    
    #set root logger to warning
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    if project_namespace is not None:
        logger = logging.getLogger(project_namespace) 
        logger.setLevel(log_level)

    if logger.handlers:
        return logger

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_stdout_handler = logging.StreamHandler(sys.stdout)
    console_stdout_handler.setLevel(logging.DEBUG)
    console_stdout_handler.setFormatter(log_format)
    logger.addHandler(console_stdout_handler)

    console_stderr_handler = logging.StreamHandler(sys.stderr)
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
    return logger