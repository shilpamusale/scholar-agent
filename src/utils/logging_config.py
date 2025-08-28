import logging
import os
from datetime import datetime

def setup_logging(
    logger_name: str, 
    log_file_prefix: str
):
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


    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent the root logger from handling messages from this logger
    logger.propagate = False

    # Avoid adding handlers if they already exist
    if logger.hasHandlers():
        return logger
    
    # Create a file handler
    log_filename = f"{log_file_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    file_handler = logging.FileHandler(os.path.join("logs", log_filename))
    file_handler.setLevel(logging.INFO)

    # Create a console handler for real-time output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # CReate a formatter and add to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s  - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



    