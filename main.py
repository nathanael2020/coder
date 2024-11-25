from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.generator import generate_code
from core.executor import execute_code, execute_solution
from core.modifier import modify_code
from utils.logger import setup_logging
from core.code_mapper import CodeMapper
from core.enhanced_debugger import EnhancedDebugger
from core.version_control import VersionControl
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils.config import logger
import traceback
import logging
from typing import Optional
import re

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

code_mapper = CodeMapper(root_directory=".")  # Map the entire current project
debugger = EnhancedDebugger(code_mapper)
version_control = VersionControl()

class CodeGenerationRequest(BaseModel):
    request: str

class CodeExecutionRequest(BaseModel):
    code: str
    timeout: Optional[int] = 10  # Default 10 seconds, None for no timeout

class CodeModificationRequest(BaseModel):
    code: str
    modification: str

@app.post("/generate/")
async def generate(request: CodeGenerationRequest):
    logger.info(f"User requested code generation: {request.request}")

    # Strip comments from the request
    request.request = re.sub(r'#.*', '', request.request)

    try:
        generated_code = generate_code(request.request)
        # After generation, update the code mapping
        code_mapper.map_codebase()
        return {"code": generated_code}
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute/")
async def execute(request: CodeExecutionRequest):
    logger.info("Executing provided code")
    execution_logs = []
    
    try:
        result = execute_code(request.code, request.timeout)
        
        if not result:
            return {"error": "Execution failed - no result returned", "logs": execution_logs}
            
        if result.get("success", False):
            return {"output": result.get("output", "No output"), "logs": execution_logs}
        else:
            # Extract error message and generate debug plan
            error_details = result.get("error", {})
            if isinstance(error_details, dict):
                error_msg = error_details.get("error", str(error_details))
            else:
                error_msg = str(error_details)
            
            logger.info("Generating debug plan")
            debug_plan = debugger.parse_error(request.code, error_msg)
            
            # Execute solution if debug plan is available
            if debug_plan:
                logger.info("Executing solution based on debug plan")
                solution_result = await execute_solution(debug_plan, request.code)
                
                if solution_result.get("success"):
                    # Try executing the code again after solution
                    logger.info("Retrying code execution after solution")
                    new_result = execute_code(request.code, request.timeout)
                    
                    if new_result.get("success"):
                        return {
                            "output": new_result.get("output"),
                            "logs": execution_logs,
                            "debug_plan": debug_plan,
                            "solution_result": solution_result
                        }
                
                # Return results even if solution didn't succeed
                return {
                    "error": error_details,
                    "debug_plan": debug_plan,
                    "solution_result": solution_result,
                    "logs": execution_logs
                }
            
            # Return error and debug plan if no solution was possible
            return {
                "error": error_details,
                "debug_plan": debug_plan,
                "logs": execution_logs
            }
            
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return {
            "error": str(e),
            "logs": execution_logs,
            "traceback": traceback.format_exc()
        }

@app.post("/modify/")
async def modify(request: CodeModificationRequest):
    logger.info(f"User requested code modification: {request.modification}")

    # Strip comments from the modification
    request.modification = re.sub(r'#.*', '', request.modification)

    try:
        # Create a new branch before modifying code
        branch_name = f"modify_{request.modification[:10]}"
        version_control.create_branch(branch_name)
        
        # Modify code and commit the changes
        modified_code = modify_code(request.code, request.modification)
        version_control.commit_changes(f"Modified code: {request.modification}")
        
        # Update code mapping after modification
        code_mapper.map_codebase()

        return {"modified_code": modified_code}
    except Exception as e:
        logger.error(f"Error modifying code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
