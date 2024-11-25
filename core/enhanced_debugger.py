import json
import re
from core.code_mapper import CodeMapper
from utils.config import logger, SecurityError
import traceback
import subprocess
import sys
from utils.config import client
from utils.sanitizer import clean_json


class EnhancedDebugger:

    def __init__(self, code_mapper):
        self.code_mapper = code_mapper

    def parse_error(self, code, error_message):
        """Parse error message and generate debug plan."""
        if not error_message:
            return None

        # Extract error type and details
        if "ModuleNotFoundError: No module named" in error_message:
            # Extract the missing module name
            module_match = re.search(r"No module named '([^']+)'", error_message)
            if module_match:
                missing_module = module_match.group(1)
                return {
                    "error_type": "missing_package",
                    "missing_module": missing_module,
                    "root_cause": f"Missing required package: {missing_module}",
                    "action_required": "package_install"
                }
        
        # ... rest of your error parsing logic ...

    def parse_error_context(self, code, error_message):
        """Extract relevant code context around the error line if applicable."""
        try:
            # Look for line number in common Python error formats
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                error_line = int(line_match.group(1))
                code_lines = code.split('\n')
                
                # Get context (50 lines before and after error)
                start_line = max(0, error_line - 50)
                end_line = min(len(code_lines), error_line + 50)
                
                return {
                    "has_code_context": True,
                    "error_line": error_line,
                    "code_context": '\n'.join(code_lines[start_line:end_line]),
                    "context_start_line": start_line + 1  # Convert to 1-based indexing
                }
        except Exception as e:
            logger.error(f"Error parsing error context: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            print(f"Error message:\n{str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
        
        return {"has_code_context": False}


    def get_debug_plan(self, code, error_details):
        """Generate structured debugging plan based on error type."""
        logger.info("Generating debug plan")
        error_info = json.loads(error_details) if isinstance(error_details, str) else error_details
        error_message = error_info.get("error", "")
        
        logger.debug(f"Error message:\n{error_message}")
        
        # Check if it's a dependency-related error
        is_import_error = any(x in error_message for x in ["ModuleNotFoundError", "ImportError"])
        logger.info(f"Error type: {'import_error' if is_import_error else 'code_error'}")
        
        if is_import_error:
            system_prompt = """You are a Python debugging expert. Analyze the error and provide specific debugging guidance. The json will be machine read, so do not include any other text.
            Provide response in this JSON format:
            {
                "error_type": "import_error",
                "missing_package": "string",
                "package_installation": {
                    "required": true,
                    "packages": ["string"],
                    "commands": ["string"]
                },
                "verification_steps": ["string"]
            }
            
            Respond only with the JSON object inside ```json and ```. Beginning with ```json and ending with ```. Do not include any other text. Do not include comments."""

        else:
            # Get code context if available
            context = self.parse_error_context(code, error_message)
            
            if context["has_code_context"]:
                system_prompt = """You are a Python debugging expert. Analyze the error and provide specific debugging guidance. The json will be machine read, so do not include any other text.
                Provide a debugging plan in this JSON format:
                {
                    "error_type": "string",
                    "error_line": number,
                    "root_cause": "string",
                    "suggested_fixes": ["string"],
                    "verification_steps": ["string"]
                }
                Respond only with the JSON object inside ```json and ```. Beginning with ```json and ending with ```. Do not include any other text. Do not include comments."""
            else:
                system_prompt = """You are a Python debugging expert. Analyze the error and provide specific debugging guidance. The json will be machine read, so do not include any other text.
                Provide response in this JSON format:
                {
                    "error_type": "string",
                    "root_cause": "string",
                    "suggested_fixes": ["string"],
                    "verification_steps": ["string"]
                }
                
                Respond only with the JSON object inside ```json and ```. Beginning with ```json and ending with ```. Do not include any other text. Do not include comments."""

        # try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Error details:\n{error_message}"}
        ]
        
        # Add code context if available
        if not is_import_error and context.get("has_code_context"):
            messages.append({
                "role": "user",
                "content": f"Code context around error (lines {context['context_start_line']}-{context['context_start_line'] + len(context['code_context'].split(chr(10))) - 1}):\n{context['code_context']}"
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
        )
        
        print(f"Response:\n{response.choices[0].message.content}")

        logger.info("Parsing debug plan")

        debug_plan = json.loads(clean_json(response.choices[0].message.content))
        
        logger.info(f"Debug plan generated:\n{debug_plan}")
        # Handle package installation if needed
        if is_import_error and debug_plan.get("package_installation", {}).get("required", False):
            packages = debug_plan["package_installation"]["packages"]
            logger.info(f"Missing packages detected: {', '.join(packages)}")
            
            approval = input("Would you like to install these packages? (y/n): ")
            logger.info(f"User approval for package installation: {approval}")
            
            if approval.lower() == 'y':
                for package in packages:
                    if not all(c.isalnum() or c in "-_." for c in package):
                        logger.error(f"Invalid package name detected: {package}")
                        raise SecurityError(f"Invalid package name: {package}")
                    try:
                        logger.info(f"Installing package: {package}")
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        logger.info(f"Successfully installed {package}")
                        logger.debug(f"Installation output:\n{result.stdout}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to install {package}: {e.stderr}")
                        debug_plan["verification_steps"].append(f"Manual installation required for {package}")
            else:
                logger.info("User declined package installation")
                debug_plan["verification_steps"].append("User declined package installation")
        
        return debug_plan
        
        # except Exception as e:
        #     logger.error(f"Error in debug plan generation: {str(e)}")
        #     logger.error(f"Traceback:\n{traceback.format_exc()}")
        #     return {
        #         "error_type": "unknown",
        #         "root_cause": str(e),
        #         "suggested_fixes": [],
        #         "verification_steps": []
        #     }

    def debug_and_fix(self, code, error_message):
        """Attempt to debug and fix the code."""
        error_details = self.parse_error(code, error_message)
        # Use an LLM to suggest changes or fix the code
        # LLM logic goes here, similar to the code modification in `modifier.py`
        return error_details
