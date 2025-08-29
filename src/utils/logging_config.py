# src/utils/logging_config.py

import logging
import os
from datetime import datetime
import configs.settings as settings # Import settings

def setup_logging(logger_name: str, log_file_prefix: str):
    """
    Sets up a logger that writes to a timestamped file in the project's logs/ directory.
    """
    # Ensure the logs directory exists using the absolute path from settings
    os.makedirs(settings.LOGS_PATH, exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        return logger

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{log_file_prefix}_{timestamp}.log"
    
    # Use the absolute path for the log file
    file_handler = logging.FileHandler(os.path.join(settings.LOGS_PATH, log_filename))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
        
    return logger
