"""
generator.py

Generator module for code generation from natural language requests.

This module provides functionality to generate Python code from user requests using
OpenAI's GPT models. It handles the communication with the OpenAI API, processes
the responses, and ensures proper error handling.

Attributes:
    None

Dependencies:
    - openai: For API communication with OpenAI
    - utils.sanitizer: For cleaning generated code
    - utils.config: For client configuration and logging
"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: core/generator.py

import traceback
from openai import OpenAI
import os
from utils.sanitizer import clean_code
from utils.config import client, logger

def generate_code(user_request):
    """Generate Python code from a natural language request using OpenAI's GPT model.

    This function takes a natural language description of desired functionality and
    generates corresponding Python code using OpenAI's API. It includes error handling,
    logging, and code sanitization.

    Args:
        user_request (str): A natural language description of the code to be generated.
            Example: "create a function that sorts a list of numbers"

    Returns:
        str: The generated Python code, cleaned and formatted.

    Raises:
        RuntimeError: If code generation fails for any reason (API errors, invalid responses, etc.).
            The error message will include the original exception details.

    Example:
        >>> code = generate_code("create a function to calculate fibonacci numbers")
        >>> print(code)
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    system_prompt = """You are an expert Python programmer. Generate clean, efficient, and safe code.
    Include proper error handling and input validation. Only respond with executable code wrapped in ```python and ```.
    
    Never include any other text than the code."""

    prompt = f"Write Python code to {user_request}. Respond with code only, wrapped in ```python and ```. Do not include any other text than the code."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to the latest model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7,
        )

        print(f"Generated code: {response.choices[0].message.content}")
        logger.info(f"Generated code: {response.choices[0].message.content}")

        return clean_code(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"Code generation failed: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Code generation failed: {str(e)}")
