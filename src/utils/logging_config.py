# Copyright 2025 Shilpa Musale
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
logging_config.py: Centralized logging configuration for the application.
"""

import logging
import os

from rich.logging import RichHandler

from configs import settings


def setup_logging(name: str) -> logging.Logger:
    """
    Sets up a logger with a rich handler for beautiful, configurable output.

    Args:
        name (str): The name of the logger (usually __name__).

    Returns:
        logging.Logger: A configured logger instance.
    """
    # --- CHANGE: Always get the level from the environment. Default to INFO if not set. ---
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Logger is already configured, don't add duplicate handlers
        return logger

    logger.propagate = False  # Prevent messages from being passed to the root logger
    logger.setLevel(log_level)

    # Ensure the logs directory exists
    settings.LOGS_PATH.mkdir(exist_ok=True)

    # --- CHANGE: Pass the log_level directly to the RichHandler ---
    # A rich handler for readable console output
    console_handler = RichHandler(level=log_level, markup=True, show_path=False, rich_tracebacks=True)

    # A file handler to save logs to a file
    file_handler = logging.FileHandler(settings.LOGS_PATH / "scholar_agent.log")
    file_handler.setLevel(logging.DEBUG)  # Always log debug level to file

    # Create formatters and add them to the handlers
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
