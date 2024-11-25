from openai import OpenAI
from dotenv import load_dotenv
import os
import traceback

from utils.logger import setup_logging

logger = setup_logging()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

class SecurityError(Exception):
    """Exception raised for security issues."""
    pass
