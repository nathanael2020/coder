"""

run_console.py

This module contains a console version of the API. It includes a main function to handle user requests and interact with the code generation, execution, and modification processes.
The console version provides a simple text-based interface for users to generate, run, and modify code interactively.
It is an alternative to the FastAPI version of the API.

"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: run_console.py

import argparse
from core.generator import generate_code
from core.executor import execute_code, execute_solution
from core.code_modifier import modify_code
from core.code_mapper import CodeMapper
from core.enhanced_debugger import EnhancedDebugger
from core.version_control import VersionControl
from utils.logger import setup_logging
import json
import os
import traceback
from utils.config import logger

# Main Interaction Loop
def main():
    logger.info("Starting Interactive Code Assistant")

    code_mapper = CodeMapper(root_directory=".")  # Map the entire current project
    debugger = EnhancedDebugger(code_mapper)
    version_control = VersionControl()

    generated_code = None  # Track if we have generated code

    while True:
        # Main menu
        print("\nWhat would you like to do?")
        options = ["1. Generate new code"]
        if generated_code:  # Only show options if we have code
            options.extend([
                "2. Run existing code",
                "3. Modify existing code"
            ])
        options.append(f"{len(options) + 1}. Quit")

        print("\n".join(options))
        choice = input("\nChoice: ")
        logger.info(f"Main menu choice: {choice}")

        try:
            if choice == '1':  # Generate new code
                user_input = input("\nWhat would you like me to create? ")
                logger.info(f"User request: {user_input}")

                generated_code = generate_code(user_input)

                os.makedirs("generated_code", exist_ok=True)
                with open(os.path.join("generated_code", "generated_code.py"), "w") as code_file:
                    code_file.write(generated_code)

                # Update code mapping after generating new code
                code_mapper.map_codebase()

                # Create a new branch for the new code
                branch_name = f"generate_{user_input[:10].replace(' ', '_')}"
                version_control.create_branch(branch_name)
                version_control.commit_changes(f"Generated code for request: {user_input}")

                print(f"\nGenerated Code:\n{generated_code}\n")
                logger.info("Code generated successfully")

            elif choice == '2' and generated_code:  # Run existing code
                logger.info("Executing existing code")
                execution_result = execute_code(generated_code, logger)
                logger.info(f"Execution result: {execution_result}")

                if execution_result["success"] == "timeout":
                    print(f"\n{execution_result['message']}")
                    if input("\nWas this the expected behavior? (y/n): ").lower() != 'y':
                        print("\nReturning to main menu to modify or regenerate code.")
                        continue
                    else:
                        print("\nExecution completed as expected.")

                elif not execution_result["success"]:
                    debug_plan = execution_result.get("debug_plan")
                    if debug_plan:
                        print("\nExecution failed. Debug Plan:")
                        print(json.dumps(debug_plan, indent=2))
                        logger.info(f"Debug plan generated: {debug_plan}")

                        if input("\nWould you like me to apply these fixes? (y/n): ").lower() == 'y':
                            solution_result = execute_solution(debug_plan, generated_code, logger)

                            if solution_result["success"]:
                                print(f"\nSuccess: {solution_result['message']}")
                                if "fixed_code" in solution_result:
                                    generated_code = solution_result["fixed_code"]
                                    print("\nFixed Code:")
                                    print(generated_code)

                                    # Save fixed code
                                    os.makedirs("generated_code", exist_ok=True)
                                    with open(os.path.join("generated_code", "generated_code.py"), "w") as code_file:
                                        code_file.write(generated_code)

                                    # Commit the fix
                                    version_control.commit_changes("Applied debug fixes to generated code")

                            else:
                                print(f"\nFailed to execute solution: {solution_result.get('error')}")
                    else:
                        print("\nNo debug plan available.")
                else:
                    print(f"\nOutput:\n{execution_result['output']}")

            elif choice == '3' and generated_code:  # Modify existing code
                try:
                    modification_request = input("\nWhat modifications would you like to make? ")
                    logger.info(f"Modification request: {modification_request}")

                    # Create a new branch for modification
                    branch_name = f"modify_{modification_request[:10].replace(' ', '_')}"
                    version_control.create_branch(branch_name)

                    modified_code = modify_code(generated_code, modification_request)
                    print(f"\nModified Code:\n{modified_code}")

                    if input("\nWould you like to keep these changes? (y/n): ").lower() == 'y':
                        generated_code = modified_code
                        os.makedirs("generated_code", exist_ok=True)
                        with open(os.path.join("generated_code", "generated_code.py"), "w") as code_file:
                            code_file.write(generated_code)

                        # Update code mapping after modification
                        code_mapper.map_codebase()

                        # Commit the changes
                        version_control.commit_changes(f"Modified code: {modification_request}")

                        print("\nChanges saved.")
                    else:
                        print("\nChanges discarded.")
                        # Optionally reset the branch if changes are discarded
                except Exception as e:
                    logger.error(f"Error modifying code: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    print(f"\nFailed to modify code: {str(e)}")
                    print(f"Traceback:\n{traceback.format_exc()}")

            elif choice == str(len(options)):  # Quit
                logger.info("User requested to quit")
                print("\nThank you for using the Interactive Code Assistant. Goodbye!")
                break

            else:
                print("\nInvalid choice. Please try again.")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            print(f"An error occurred: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
if __name__ == "__main__":
    main()
