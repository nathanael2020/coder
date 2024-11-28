"""
config.py

This module contains the configuration for the API. It includes the OpenAI client, the logging setup, and the security error.

Key Components:
    - OpenAI client configuration
    - Environment variable loading
    - Logging setup
    - Security error definition

Dependencies:
    - openai: For API client
    - python-dotenv: For environment variable management
    - logging: For application logging
"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: utils/config.py

from openai import OpenAI
from dotenv import load_dotenv
import os
import traceback

from utils.logger import setup_logging

logger = setup_logging()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))
FAISS_INDEX_BASE = os.getenv("FAISS_INDEX_BASE")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")

# Additional directories to ignore during indexing (these can remain static)
IGNORE_PATHS = [
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
    "tests",
    "logs",

]


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.openai.com/v1",
)

class SecurityError(Exception):
    """
    Custom exception for handling security-related errors in the application.
    
    This exception should be raised when security violations occur, such as:
        - Invalid authentication attempts
        - Unauthorized access to protected resources
        - Security policy violations
        
    Inherits from:
        Exception: Base Python exception class
    """
    pass
