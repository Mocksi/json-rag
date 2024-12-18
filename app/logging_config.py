import logging
import os

# Default to INFO, but allow override through environment variable
LOG_LEVEL = os.environ.get('APP_LOG_LEVEL', 'INFO').upper()

def configure_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Create a function to get logger instances with consistent configuration
def get_logger(name):
    """Get a logger instance with consistent configuration."""
    return logging.getLogger(name) 