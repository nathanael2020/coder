from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Union, List, Optional
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json
from fastapi.responses import HTMLResponse
from core.codebase_manager import CodebaseManager
from utils.config import logger
from fastapi.responses import FileResponse
from core.codebase_manager import get_codebase_manager
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import traceback
from core.code_mapper import CodeMapper
from core.code_modifier import CodeModifier

class CodebaseRequest(BaseModel):
    source: Union[HttpUrl, str]


    keep_workspace: bool = True  # Option to keep the workspace after analysis
    index_directory: str = ""  # Optional directory to index within the codebase

class ImportRequest(BaseModel):
    source: Union[HttpUrl, str]
    workspace_dir: Optional[str] = "sandbox"

class ModifyRequest(BaseModel):
    filepath: str
    modifications: List[dict]  # List of modification operations
    context: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "filepath": "src/main.py",
                "modifications": [
                    {
                        "line": 10,
                        "operation": "modify",
                        "modified_code": "def new_function():"
                    }
                ],
                "context": {
                    "description": "Optional context about the modification"
                }
            }
        }

router = APIRouter()

@router.get('/', response_class=HTMLResponse)
async def read_root():
    return FileResponse('static/index.html')


# @router.post("/analyze")
# async def analyze_codebase(request: CodebaseRequest):
#     """Analyze a codebase from git URL."""
#     codebase_manager = CodebaseManager()
    
#     logger.info(f"Importing codebase from {request.source}")
#     if not codebase_manager.import_codebase(request.source):
#         raise HTTPException(status_code=400, detail="Failed to import codebase")
    
#     # Set active directory if specified, otherwise use repo root
#     if request.index_directory:
#         logger.info(f"Setting active directory to: {request.index_directory}")
#         if not codebase_manager.set_active_directory(request.index_directory):
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid index directory: {request.index_directory}"
#             )
#     else:
#         # Use the root of the repository
#         logger.info("Using repository root as active directory")
#         if not codebase_manager.set_active_directory("."):
#             raise HTTPException(
#                 status_code=500, 
#                 detail="Failed to set repository root as active directory"
#             )
        
#     logger.info("Analyzing codebase...")
#     analysis = codebase_manager.analyze_codebase()
    
#     if not analysis:
#         raise HTTPException(status_code=500, detail="Analysis failed")
        
#     # Ensure all required fields exist
#     analysis.setdefault('files', {})
#     analysis.setdefault('code', {})
#     analysis.setdefault('git', {})
#     analysis.setdefault('llm_processing', {
#         'total_characters': 0,
#         'can_process_with_gpt4': False,
#         'concatenated_code': ''
#     })
#     analysis.setdefault('enhanced_stats', {
#         'estimated_read_time': 'N/A'
#     })
    
#     logger.info(f"Returning analysis: {analysis}")
    
#     return JSONResponse(content={
#         "status": "success",
#         "data": analysis
#     })
    # except Exception as e:
    #     logger.error(f"Error analyzing codebase: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     # Only cleanup if not keeping workspace
    #     if not request.keep_workspace:
    #         manager.cleanup(keep_workspace=False)

# @router.post("/modify")

# async def modify_code_endpoint(request: ModifyRequest):
#     try:
#         codebase_manager = get_codebase_manager()
        
#         if not codebase_manager.repo_dir:
#             raise HTTPException(
#                 status_code=400, 
#                 detail="No repository has been imported. Please import a codebase first."
#             )
            
#         code_mapper = CodeMapper(codebase_manager.repo_dir)
#         modifier = CodeModifier(code_mapper)

#         # Process modifications
#         result = modifier.modify_file(
#             request.filepath,
#             request.modifications
#         )
        
#         if not result:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Failed to modify file"
#             )
            
#         return {"status": "success"}
        
#     except Exception as e:
#         logger.error(f"Error in code modification: {str(e)}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         print(f"Error in code modification: {str(e)}")
#         print(f"Traceback: {traceback.format_exc()}")

#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_file(
    file_path: str,
    content: str,
    session_id: str
):
    """Create a new file in the workspace."""
    workspace_dir = Path("sandbox") / f"workspace_{session_id}"
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    codebase_manager = CodebaseManager()
    codebase_manager.workspace_dir = workspace_dir
    
    if not codebase_manager.create_file(file_path, content):
        raise HTTPException(status_code=500, detail="Failed to create file")
        
    return JSONResponse(content={"status": "success"})

@router.post("/modify-and-execute")
async def modify_and_execute(
    file_path: str,
    content: str,
    session_id: str
):
    """Modify a file and execute it in a Docker container."""
    workspace_dir = Path("sandbox") / f"workspace_{session_id}"
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    codebase_manager = CodebaseManager()
    codebase_manager.workspace_dir = workspace_dir
    
    result = codebase_manager.modify_and_execute(file_path, content)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
        
    return JSONResponse(content=result)