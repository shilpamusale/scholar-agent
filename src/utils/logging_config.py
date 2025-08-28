# src/utils/logging_config.py

import logging
import os
from datetime import datetime


def setup_logging(logger_name: str, log_file_prefix: str):
    """
    Sets up a logger that writes to a timestamped file in the logs/ directory.

    Args:
        logger_name (str): The name of the logger.
        log_file_prefix (str): The prefix for the log file name.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        return logger

    # Create a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{log_file_prefix}_{timestamp}.log"

    file_handler = logging.FileHandler(os.path.join("logs", log_filename))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
