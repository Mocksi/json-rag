"""
Logging Configuration Module

This module provides consistent logging configuration across the application.
It allows for log level configuration through environment variables and
ensures uniform log formatting across all modules.

Environment Variables:
    APP_LOG_LEVEL: Set the logging level (default: 'INFO')
                   Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import logging
import os

# Default to INFO, but allow override through environment variable
LOG_LEVEL = os.environ.get('APP_LOG_LEVEL', 'INFO').upper()

def configure_logging():
    """
    Configure global logging settings for the application.
    
    This function sets up:
    - Log level (configurable via APP_LOG_LEVEL environment variable)
    - Log format with timestamp, logger name, level, and message
    - Timestamp format (YYYY-MM-DD HH:MM:SS)
    
    Example:
        >>> configure_logging()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
        2023-12-18 19:15:00 - myapp - INFO - Application started
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_logger(name):
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name (str): The name for the logger, typically __name__
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    return logging.getLogger(name)