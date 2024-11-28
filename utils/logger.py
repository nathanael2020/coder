"""
logger.py

Logger Configuration Module

This module provides logging setup functionality for the application. It configures both file
and console logging with rotation capabilities to manage log file sizes.

The logger is configured to:
- Write logs to both console and file
- Rotate log files when they reach 1MB
- Maintain 5 backup log files
- Use UTF-8 encoding
- Format logs with timestamp, level, and message
"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: utils/logger.py


import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """
    Configure and initialize the application logger with both file and console handlers.
    
    Creates a rotating file handler that:
    - Maintains log files up to 1MB in size
    - Keeps 5 backup files
    - Appends to existing log files
    - Uses UTF-8 encoding
    
    Returns:
        logging.Logger: Configured logger instance with both file and console handlers,
                       formatted with timestamp, log level, and message.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure the logger
    logger = logging.getLogger('coder')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        'logs/coder.log',
        mode='a',           # 'a' for append instead of 'w' for write
        maxBytes=1024*1024, # 1MB per file
        backupCount=5,      # Keep 5 backup files
        encoding='utf-8'
    )
    
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent logging from propagating to the root logger
    logger.propagate = False
    
    return logger