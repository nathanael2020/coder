"""
executor.py

This module contains the executor functions for the API. It includes the execute_code function to execute code in a controlled environment with configurable timeout, and the execute_solution function to execute the solutions proposed in the debug plan.

"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: core/executor.py

import subprocess
import os
import platform
import logging
from utils.config import SecurityError
import sys
from utils.sanitizer import clean_code
from utils.config import client, logger
import traceback
from core.debugger import get_debug_plan
import re

# def execute_code_in_docker():
#     """Use Docker to execute the code safely in an isolated container.
    
#     This function builds and runs a Docker container to execute code in a sandboxed
#     environment, providing an additional layer of security and isolation.
    
#     Returns:
#         None
    
#     Note:
#         Requires Docker to be installed and running on the system.
#         The Dockerfile should be present in the current directory.
#     """
#     os.system("docker build -t sandbox-executor .")
#     os.system("docker run --rm sandbox-executor")


def execute_code(code, timeout=10):
    """Execute Python code in a controlled environment with configurable timeout.
    
    This function executes the provided code in a sandbox directory with safety measures
    and timeout constraints. It captures both stdout and stderr, along with detailed
    error information if execution fails.
    
    Args:
        code (str): The Python code to execute.
        timeout (int, optional): Maximum execution time in seconds. Defaults to 10.
            Set to None for no timeout.
    
    Returns:
        dict: A dictionary containing execution results with the following structure:
            On success:
                {
                    "success": True,
                    "output": str  # Standard output from the code execution
                }
            On failure:
                {
                    "success": False,
                    "error": {
                        "error": str,  # Error message
                        "error_type": str,  # Type of error (runtime_error/timeout_error)
                        "system_info": {  # System information for debugging
                            "os": str,
                            "python_version": str,
                            "working_directory": str
                        },
                        "traceback": str  # Full error traceback
                    }
                }
    
    Raises:
        OSError: If sandbox directory creation fails.
        IOError: If writing to sandbox file fails.
    """
    logger.info("Starting code execution")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sandbox_dir = os.path.join(base_dir, "../sandbox")
    
    if not os.path.exists(sandbox_dir):
        os.makedirs(sandbox_dir)
    
    sandbox_file = os.path.join(sandbox_dir, "generated_code.py")
    
    with open(sandbox_file, "w") as f:
        f.write(code)
    
    try:
        # Run with timeout only if specified
        result = subprocess.run(
            ["python", sandbox_file],
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else None,
            cwd=sandbox_dir
        )
        
        if result.returncode == 0:
            logger.info("Code execution successful")
            return {"success": True, "output": result.stdout}
        else:
            error_details = {
                "error": result.stderr,
                "error_type": "runtime_error",
                "system_info": {
                    "os": platform.system(),
                    "python_version": platform.python_version(),
                    "working_directory": os.getcwd()
                },
                "traceback": result.stderr
            }
            return {"success": False, "error": error_details}
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Code execution timed out after {timeout} seconds")
        error_details = {
            "error": "Execution timeout",
            "error_type": "timeout_error",
            "system_info": {
                "os": platform.system(),
                "python_version": platform.python_version(),
                "working_directory": os.getcwd()
            },
            "traceback": f"Code execution timed out after {timeout} seconds"
        }
        return {"success": False, "error": error_details}


async def execute_solution(debug_plan, original_code):
    """Execute the solutions proposed in the debug plan, particularly handling package installations.
    
    This asynchronous function processes the debug plan and executes necessary actions,
    currently focusing on package installation for missing dependencies. It includes
    smart handling of package name corrections and multiple installation attempts.
    
    Args:
        debug_plan (dict): A dictionary containing the debug plan with the following structure:
            {
                "error_type": str,  # Type of error (e.g., "missing_package")
                "action_required": str,  # Required action (e.g., "package_install")
                "missing_module": str  # Name of the missing package
            }
        original_code (str): The original code that generated the error (currently unused).
    
    Returns:
        dict: A dictionary containing the execution results with the following structure:
            On successful package installation:
                {
                    "success": True,
                    "action": "package_installed",
                    "package": str,  # Installed package name
                    "original_name": str  # Original package name before correction
                }
            On failed package installation:
                {
                    "success": False,
                    "error": str,  # Error message
                    "action": "package_install_failed",
                    "package": str,  # Package name
                    "error_output": str  # Detailed error output
                }
            On no matching solution:
                {
                    "success": False,
                    "message": "No matching solution found"
                }
    
    Note:
        Currently only handles package installation errors. Future versions may
        handle additional types of debugging actions.
    """
    logger.info("Starting solution execution")

    if not debug_plan:
        logger.info("No debug plan provided")
        return {"success": False, "message": "No debug plan provided"}

    # Handle package installation
    if debug_plan.get("error_type") == "missing_package" and debug_plan.get("action_required") == "package_install":
        missing_module = debug_plan.get("missing_module")
        if missing_module:
            logger.info(f"Missing package detected: {missing_module}")
            
            try:
                # First attempt to install
                logger.info(f"Attempting to install package: {missing_module}")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", missing_module],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # Extract just the package name from error messages like:
                    # "use 'scikit-learn' rather than 'sklearn'" or
                    # "use package-name rather than original-name"
                    correction_match = re.search(r"use ['`]([^'`\s]+)['`]", result.stderr)
                    if correction_match:
                        corrected_name = correction_match.group(1)
                        logger.info(f"Found package name correction: {corrected_name}")
                        
                        # Try again with corrected name
                        logger.info(f"Retrying installation with corrected package name: {corrected_name}")
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", corrected_name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        missing_module = corrected_name
                
                logger.info(f"Successfully installed {missing_module}")
                return {
                    "success": True,
                    "action": "package_installed",
                    "package": missing_module,
                    "original_name": debug_plan.get("missing_module")
                }
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {missing_module}: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "action": "package_install_failed",
                    "package": missing_module,
                    "error_output": e.output if hasattr(e, 'output') else None
                }
    
    logger.info("No solution executed")
    return {"success": False, "message": "No matching solution found"}
