import re
import json

def parse_error_context(code, error_message):
    """Extract relevant code context around the error line if applicable."""
    try:
        line_match = re.search(r'line (\d+)', error_message)
        if line_match:
            error_line = int(line_match.group(1))
            code_lines = code.split('\n')
            start_line = max(0, error_line - 50)
            end_line = min(len(code_lines), error_line + 50)
            
            return {
                "has_code_context": True,
                "error_line": error_line,
                "code_context": '\n'.join(code_lines[start_line:end_line]),
                "context_start_line": start_line + 1
            }
    except Exception:
        pass

    return {"has_code_context": False}

def get_debug_plan(code, error_details, client, logger):
    """Generate structured debugging plan based on error type."""
    logger.info("Generating debug plan")
    error_info = json.loads(error_details) if isinstance(error_details, str) else error_details
    error_message = error_info.get("error", "")
    
    is_import_error = any(x in error_message for x in ["ModuleNotFoundError", "ImportError"])
    logger.info(f"Error type: {'import_error' if is_import_error else 'code_error'}")

    system_prompt = """You are a Python debugging expert. Analyze the error and provide guidance.
    Provide response in this JSON format: { ... } Respond only with JSON inside ```json```."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Error details:\n{error_message}"}
            ],
            max_tokens=2000,
        )
        
        debug_plan = json.loads(response.choices[0].message.content)
        return debug_plan
    
    except Exception as e:
        logger.error(f"Error in debug plan generation: {str(e)}")
        return {"error_type": "unknown", "root_cause": str(e), "suggested_fixes": []}
