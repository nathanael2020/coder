"""
sanitizer.py

This module contains sanitization functions for ensuring safe code execution and JSON processing.

Key Functions:
    - clean_code: Sanitizes and validates Python code before execution
    - clean_json: Cleans and extracts valid JSON from string input

Security:
    This module implements basic security measures to prevent execution of potentially
    harmful code and ensure proper JSON formatting.
"""

import re
from utils.config import SecurityError


def clean_path(path):
    """
    Clean and sanitize a file path.
    """
    return path.replace("\\", "/")

def clean_code(code):
    """
    Sanitize and validate Python code before execution.
    
    This function performs basic security validation by:
        1. Removing Python code block markers
        2. Checking for potentially dangerous operations
        3. Stripping whitespace
    
    Args:
        code (str): Raw code string, potentially containing markdown code blocks
        
    Returns:
        str: Cleaned and validated code string
        
    Raises:
        SecurityError: If potentially unsafe operations are detected
        
    Example:
        >>> clean_code("```python\\nprint('hello')\\n```")
        "print('hello')"
    """
    code = code.replace("```python", "").replace("```", "").strip()
    # Basic security validation
    forbidden_terms = ['os.system', 'subprocess.', 'eval(', 'exec(']
    if any(term in code for term in forbidden_terms):
        raise SecurityError("Potentially unsafe code detected")
    return code


def clean_json(json_string):
    """
    Clean JSON string by removing non-JSON content and comments.
    
    This function:
        1. Extracts JSON content from markdown code blocks
        2. Removes single-line comments (// and #)
        3. Removes multi-line comments (/* ... */)
    
    Args:
        json_string (str): Raw JSON string, potentially containing comments and markdown
        
    Returns:
        str: Cleaned JSON string ready for parsing
        
    Example:
        >>> clean_json("```json\\n{\\n  // comment\\n  "key": "value"\\n}\\n```")
        '\\n{\\n  "key": "value"\\n}\\n'
    """
    json_str = re.search(r'```json(.*?)```', json_string, re.DOTALL).group(1)
    # Remove single-line comments (both // and # styles)
    json_str = re.sub(r'//.*$|#.*$', '', json_str, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    return json_str
    
