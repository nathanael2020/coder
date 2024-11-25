import json
import re
from core.code_mapper import CodeMapper

class EnhancedDebugger:
    def __init__(self, code_mapper):
        self.code_mapper = code_mapper

    def parse_error(self, code, error_message):
        """Extract error context and provide structured debugging suggestions."""
        line_match = re.search(r'line (\d+)', error_message)
        if line_match:
            error_line = int(line_match.group(1))
            code_lines = code.split('\n')
            start_line = max(0, error_line - 5)
            end_line = min(len(code_lines), error_line + 5)
            context = "\n".join(code_lines[start_line:end_line])
            return {
                "error_line": error_line,
                "code_context": context,
                "suggestions": [
                    "Check for incorrect variable names or missing imports.",
                    "Ensure that the function or variable used is defined properly."
                ]
            }
        return {"error": "Unable to parse the error message"}

    def debug_and_fix(self, code, error_message):
        """Attempt to debug and fix the code."""
        error_details = self.parse_error(code, error_message)
        # Use an LLM to suggest changes or fix the code
        # LLM logic goes here, similar to the code modification in `modifier.py`
        return error_details
