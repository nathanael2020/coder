from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.generator import generate_code
from core.executor import execute_code
from core.modifier import modify_code
from utils.logger import setup_logging
from core.code_mapper import CodeMapper
from core.enhanced_debugger import EnhancedDebugger
from core.version_control import VersionControl

app = FastAPI()
logger = setup_logging()
code_mapper = CodeMapper(root_directory=".")  # Map the entire current project
debugger = EnhancedDebugger(code_mapper)
version_control = VersionControl()

class CodeGenerationRequest(BaseModel):
    request: str

class CodeExecutionRequest(BaseModel):
    code: str

class CodeModificationRequest(BaseModel):
    code: str
    modification: str

@app.post("/generate/")
async def generate(request: CodeGenerationRequest):
    logger.info(f"User requested code generation: {request.request}")
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
    try:
        result = execute_code(request.code, logger)
        if result["success"]:
            return {"output": result["output"]}
        else:
            # In case of an error, generate a debug plan and return it
            error_details = debugger.parse_error(request.code, result["error"])
            return {"error": result["error"], "debug_plan": error_details}
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/modify/")
async def modify(request: CodeModificationRequest):
    logger.info(f"User requested code modification: {request.modification}")
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
