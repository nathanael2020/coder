import subprocess
import os
import platform
import logging
from utils.config import SecurityError
import sys
from utils.sanitizer import clean_code
from utils.config import client
import traceback
from core.debugger import get_debug_plan

def execute_code_in_docker():
    """Use Docker to execute the code safely."""
    os.system("docker build -t sandbox-executor .")
    os.system("docker run --rm sandbox-executor")


def execute_code(code, logger):
    """Execute code in a controlled environment with proper error handling."""
    logger.info("Starting code execution")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sandbox_dir = os.path.join(base_dir, "../sandbox")
    
    if not os.path.exists(sandbox_dir):
        logger.info(f"Creating sandbox directory: {sandbox_dir}")
        os.makedirs(sandbox_dir)
    
    sandbox_file = os.path.join(sandbox_dir, "generated_code.py")

    with open(sandbox_file, "w") as code_file:
        code_file.write(code)
    logger.info(f"Code written to sandbox file: {sandbox_file}")

    try:
        result = subprocess.run(
            ["python", sandbox_file],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=sandbox_dir
        )
    
        if result.returncode == 0:
            logger.info("Code execution successful")
            return {"success": True, "output": result.stdout}
    
    except subprocess.TimeoutExpired:
        logger.info("Code execution timed out after 10 seconds")
        logger.error(f"Code execution failed with error:\n{result.stderr}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return {"success": "timeout", "message": "Code execution timed out after 10 seconds"}
    
    except Exception as e:

        error_details = {
            "error": result.stderr,
            "error_type": "runtime_error" if "ModuleNotFoundError" not in result.stderr else "import_error",
            "system_info": {
                "os": platform.system(),
                "python_version": platform.python_version(),
                "working_directory": os.getcwd()
            },
            "traceback": traceback.format_exc(),
            "message": "An error occurred during code execution: {e}"

        }

        debug_plan = get_debug_plan(code, error_details, logger)

        logger.info(f"Debug plan: {debug_plan}")
        logger.info(f"Error details: {error_details}")
        logger.info(f"Traceback: {traceback.format_exc()}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"Error details: {error_details}")
        print(f"Debug plan: {debug_plan}")

        return {"success": False, "error": error_details, "debug_plan": debug_plan}

    # except subprocess.TimeoutExpired:
    #     logger.info("Code execution timed out after 10 seconds")
    #     return {"success": "timeout", "message": "Code execution timed out after 10 seconds"}
    
    # except Exception as e:
    #     logger.error(f"Exception during code execution: {str(e)}")
    #     return {"success": False, "error": str(e)}
    
    # finally:
    #     if os.path.exists(sandbox_file):
    #         logger.info("Cleaning up sandbox file")
    #         os.remove(sandbox_file)


# def execute_solution(debug_plan, original_code, logger):
#     """Execute the solutions proposed in the debug plan."""
#     logger.info("Starting solution execution")
    
#     if debug_plan.get("package_installation", {}).get("required", False):
#         packages = debug_plan["package_installation"]["packages"]
#         logger.info(f"Installing packages: {packages}")
        
#         for package in packages:
#             if not all(c.isalnum() or c in "-_." for c in package):
#                 logger.error(f"Invalid package name: {package}")
#                 raise SecurityError(f"Invalid package name: {package}")
            
#             try:
#                 logger.info(f"Running pip install for {package}")
#                 result = subprocess.run(
#                     [sys.executable, "-m", "pip", "install", package],
#                     check=True,
#                     capture_output=True,
#                     text=True
#                 )
#                 logger.info(f"Successfully installed {package}")
#                 return {"success": True, "message": f"Successfully installed {package}"}
#             except subprocess.CalledProcessError as e:
#                 logger.error(f"Failed to install {package}: {e.stderr}")
#                 return {"success": False, "error": f"Failed to install {package}: {e.stderr}"}
    
#     elif debug_plan.get("suggested_fixes"):
#         # For code fixes, we'll need to generate the fixed code using the LLM
#         logger.info("Generating code fixes")
#         fixes = "\n".join(debug_plan["suggested_fixes"])
        
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a Python code fixing expert. Apply the suggested fixes to the code."},
#                     {"role": "user", "content": f"Original code:\n{original_code}\n\nApply these fixes:\n{fixes}"}
#                 ],
#                 max_tokens=2000
#             )
            
#             fixed_code = clean_code(response.choices[0].message.content)
#             logger.info("Successfully generated fixed code")
#             return {
#                 "success": True,
#                 "fixed_code": fixed_code,
#                 "message": "Code has been fixed according to suggestions"
#             }
#         except Exception as e:
#             logger.error(f"Failed to generate fixed code: {str(e)}")
#             return {"success": False, "error": f"Failed to generate fixed code: {str(e)}"}
    
#     return {"success": False, "error": "No actionable solutions found in debug plan"}



import subprocess
import sys
import os

def execute_solution(debug_plan, original_code, logger):
    """Execute the solutions proposed in the debug plan."""
    logger.info("Starting solution execution")

    # Handle package installation if needed
    if debug_plan.get("package_installation", {}).get("required", False):
        packages = debug_plan["package_installation"]["packages"]
        install_commands = debug_plan["package_installation"]["commands"]

        logger.info(f"Missing packages detected: {', '.join(packages)}")
        logger.info(f"Suggested commands: {install_commands}")

        # Ask for user permission before installing
        approval = input("\nThe following packages are required: "
                         f"{', '.join(packages)}.\n"
                         f"To install them, we need to run the following command(s):\n"
                         f"{os.linesep.join(install_commands)}\n"
                         "These commands may require root privileges. "
                         "Do you want to proceed? (y/n): ")

        if approval.lower() == 'y':
            # Execute system-level commands
            for command in install_commands:
                try:
                    # Ensure command runs with elevated privileges if needed
                    logger.info(f"Running command: {command}")
                    result = subprocess.run(
                        command,
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f"Successfully ran command: {command}")
                    logger.debug(f"Command output:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to run command '{command}': {e.stderr}")
                    return {"success": False, "error": f"Failed to run command '{command}': {e.stderr}"}
        else:
            logger.info("User declined package installation.")
            return {"success": False, "error": "User declined package installation."}

    # Handle other debug steps if available
    if debug_plan.get("suggested_fixes"):
        logger.info("Generating code fixes")
        fixes = "\n".join(debug_plan["suggested_fixes"])

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Python code fixing expert. Apply the suggested fixes to the code."},
                    {"role": "user", "content": f"Original code:\n{original_code}\n\nApply these fixes:\n{fixes}"}
                ],
                max_tokens=2000
            )
            fixed_code = clean_code(response.choices[0].message.content)
            logger.info("Successfully generated fixed code")
            return {
                "success": True,
                "fixed_code": fixed_code,
                "message": "Code has been fixed according to suggestions"
            }
        except Exception as e:
            logger.error(f"Failed to generate fixed code: {str(e)}")
            return {"success": False, "error": f"Failed to generate fixed code: {str(e)}"}

    return {"success": False, "error": "No actionable solutions found in debug plan"}
