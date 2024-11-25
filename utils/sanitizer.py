import re
from utils.config import SecurityError

def clean_code(code):
    """Sanitize and validate code before execution."""
    code = code.replace("```python", "").replace("```", "").strip()
    # Basic security validation
    forbidden_terms = ['os.system', 'subprocess.', 'eval(', 'exec(']
    if any(term in code for term in forbidden_terms):
        raise SecurityError("Potentially unsafe code detected")
    return code


def clean_json(json_string):
    """Clean JSON string by removing any non-JSON content."""
    json_str = re.search(r'```json(.*?)```', json_string, re.DOTALL).group(1)
    # Remove single-line comments (both // and # styles)
    json_str = re.sub(r'//.*$|#.*$', '', json_str, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    return json_str
    
