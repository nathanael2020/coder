import logging
import os
import platform
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path



# Setup logging configuration
def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create logs directory if it doesn't exist
    # Get project root directory (2 levels up from utils folder)
    PROJECT_ROOT = Path(__file__).parent.parent

    # Create logs directory if it doesn't exist
    LOGS_DIR = PROJECT_ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamp-based log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"coder_{timestamp}.log")
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Log system information at startup
    logging.info("=== Session Started ===")
    logging.info(f"Python Version: {platform.python_version()}")
    logging.info(f"Operating System: {platform.system()} {platform.version()}")
    logging.info(f"Working Directory: {os.getcwd()}")
    
    return logging.getLogger(__name__)
