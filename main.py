"""
main.py

This module is the entry point for the API. It defines the FastAPI app and the endpoints 
for code generation, execution, and modification. Importantly, it also initializes the 
code mapper, debugger, version control, and other core components.

It is an alternative to run_console.py, providing a REST API for the API.

Components:
    - FastAPI app initialization and middleware setup
    - Static file serving
    - Core service initialization (CodeMapper, EnhancedDebugger, VersionControl)
    - Data models for requests/responses
    - API endpoints for code generation, execution and modification
"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.generator import generate_code
from core.executor import execute_code, execute_solution
from core.code_modifier import CodeModifier
from core.code_mapper import CodeMapper
from core.enhanced_debugger import EnhancedDebugger
from core.version_control import VersionControl
from core.codebase_routes import router as codebase_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils.config import logger
import traceback
from typing import Optional
import re
from openai import OpenAI
from utils.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from utils.search import search_code
from typing import List, Dict, Optional
from utils.index import IndexManager
from core.codebase_manager import get_codebase_manager
from utils.sanitizer import clean_code
import os
from custom_mapper2 import get_codebase_metadata, condense_codebase_metadata
import json
app = FastAPI()
from utils.sanitizer import clean_json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from core.codebase_routes import ImportRequest
from fastapi.responses import JSONResponse
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

# Include codebase routes
app.include_router(codebase_router, prefix="/api/codebase", tags=["codebase"])


code_mapper = CodeMapper(".")  # Map the entire current project
debugger = EnhancedDebugger(code_mapper)
version_control = VersionControl()

class CodeGenerationRequest(BaseModel):
    """
    Data model for code generation requests.
    
    Attributes:
        request (str): The natural language request describing the code to be generated
    """
    request: str

class CodeExecutionRequest(BaseModel):
    """
    Data model for code execution requests.
    
    Attributes:
        code (str): The code to be executed
        timeout (Optional[int]): Maximum execution time in seconds, defaults to 10
    """
    code: str
    timeout: Optional[int] = 10

class CodeModificationRequest(BaseModel):
    """
    Data model for code modification requests.
    
    Attributes:
        code (str): The original code to be modified
        modification (str): The natural language description of desired modifications
    """
    code: str
    modification: str

class RAGQueryRequest(BaseModel):
    """
    Data model for RAG-based code queries.
    
    Attributes:
        query (str): The natural language query about the codebase
    """
    query: str

SYSTEM_PROMPT = """
You are an expert coding assistant. Your task is to help users with their question. Use the retrieved code context to inform your responses, but feel free to suggest better solutions if appropriate.
"""

PRE_PROMPT = """
Based on the user's query and the following code context, provide a helpful response. If improvements can be made, suggest them with explanations.

User Query: {query}

Retrieved Code Context:
{code_context}

Your response:
"""

client = OpenAI(api_key=OPENAI_API_KEY)
# model = OPENAI_CHAT_MODEL
model = "gpt-4o-mini"

sandbox_root = Path("sandbox")
sandbox_root.mkdir(exist_ok=True)

# Create a unique workspace for this session
workspace_dir = sandbox_root / "workspace"
workspace_dir.mkdir(parents=True, exist_ok=True)

codebase_manager = None

def execute_rag_flow(user_query: str) -> str:
    try:
        # Perform code search
        search_results = search_code(user_query)
        
        if not search_results:
            return "No relevant code found for your query."
        
        # Prepare code context
        code_context = "\n\n".join([
            f"File: {result['filename']}\n{result['content']}"
            for result in search_results[:3]  # Limit to top 3 results
        ])
        
        # Construct the full prompt
        full_prompt = PRE_PROMPT.format(query=user_query, code_context=code_context)
        
        logger.info(f"Full prompt: {full_prompt}")

        # Generate response using OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Error in RAG flow execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/")
async def generate(request: CodeGenerationRequest):
    """
    Endpoint to generate code based on a natural language request.
    
    Args:
        request (CodeGenerationRequest): The generation request containing the description
        
    Returns:
        dict: Contains the generated code
        
    Raises:
        HTTPException: If code generation fails
    """
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
    """
    Endpoint to execute provided code with error handling and debugging support.
    
    Args:
        request (CodeExecutionRequest): The execution request containing code and timeout
        
    Returns:
        dict: Contains execution output or error details, debug plan, and execution logs
    """
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

class ModifyRequest(BaseModel):
    """Request model for code modifications."""
    instruction: str
    currentCode: Optional[str] = None
    filepath: Optional[str] = None
    context: Optional[str] = None

MODIFY_SYSTEM_PROMPT = """
You are an expert coding assistant. Using the provided code context and file content, 
help modify the code according to the user's instruction. Focus on the specific file 
and location mentioned in the context. Provide clear, concise code changes.
"""

MODIFY_PROMPT = """
Based on the following context and instruction, provide code modifications:

File to Modify: {filepath}

Relevant Code Context:
{code_context}

Current File Content:
{file_content}

Instruction: {instruction}

Please provide the necessary code changes to implement this modification.
"""

@app.post("/modify/")
async def modify_code_endpoint(request: ModifyRequest):
    # Get the manager instance and log its state
    manager = get_codebase_manager()
        # Initialize CodeMapper with the imported repository path
    code_mapper = CodeMapper(manager.repo_dir)
    modifier = CodeModifier(code_mapper)

    logger.info(f"CodebaseManager state - repo_dir: {manager.repo_dir}")
    
    # try:

    modified_code = modifier.modify_code(request.instruction, request.currentCode, request.filepath, request.context)

    return {"modified_code": modified_code}

    # except Exception as e:
    #     logger.error(f"Error modifying code: {str(e)}")
    #     raise HTTPException(status_code=500, detail=str(e))


# @app.post("/query/")
# async def query_codebase(request: RAGQueryRequest):
#     logger.info(f"Processing RAG query: {request.query}")
    
#     codebase_manager = get_codebase_manager()
#     logger.info(f"Query endpoint - manager state: {vars(codebase_manager)}")
    
#     if not codebase_manager.repo_dir:
#         raise HTTPException(
#             status_code=400, 
#             detail="No repository has been imported yet"
#         )
    
#     # Get index manager instance
#     index_manager = codebase_manager.get_index_manager()
#     logger.info(f"Retrieved index manager: {index_manager}")
    
#     if not index_manager:
#         raise HTTPException(
#             status_code=500, 
#             detail="Failed to initialize index manager"
#         )
    
@app.post("/query/")
async def query_codebase(request: RAGQueryRequest):
    """
    Endpoint to query the codebase using RAG retrieval and LLM processing.
    
    Args:
        request (RAGQueryRequest): The query request containing the natural language question
        
    Returns:
        dict: Contains the AI response based on retrieved code context
        
    Raises:
        HTTPException: If query processing fails
    """
    logger.info(f"Processing RAG query: {request.query}")
    
    codebase_manager = get_codebase_manager()
    # try:
    # Get index manager instance
    # Get index manager instance with proper error handling
    index_manager = codebase_manager.get_index_manager()
    if not index_manager:
        logger.error("Failed to get IndexManager instance")
        raise HTTPException(status_code=500, detail="Failed to initialize index manager")
    
    logger.info(f"Got IndexManager instance: {index_manager}")
        
    # Ensure index is loaded
    if not index_manager.is_index_loaded():
        logger.info("Index not loaded, attempting to load...")
        index_manager.load_index()
    
    # Get metadata for context (optional, but useful for debugging)
    metadata = index_manager.get_metadata()
    logger.info(f"Using index with metadata: {metadata}")
    
    # Now execute the RAG flow with loaded index
    response = execute_rag_flow(request.query)
    return {"response": response}
    # except Exception as e:
    #     logger.error(f"Error processing query: {str(e)}")
    #     raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata/")
async def get_metadata():
    """
    Endpoint to retrieve the current metadata from the index.
    
    Returns:
        dict: Contains the metadata from the index
    """
    try:
        index_manager = codebase_manager.get_index_manager()
        metadata = index_manager.get_metadata()
        return {"metadata": metadata}
    except Exception as e:
        logger.error(f"Error retrieving metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-index/")
async def load_index():
    """
    Endpoint to load the index from disk.
    
    Returns:
        dict: Status message
    """
    try:
        index_manager = codebase_manager.get_index_manager()
        index_manager.load_index()
        return {"status": "Index loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Modify your startup event to initialize the index manager
@app.on_event("startup")
async def startup_event():
    """Initialize the index manager and other startup tasks."""
    try:
        # Get the sandbox directory path
        sandbox_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sandbox')
        os.makedirs(sandbox_dir, exist_ok=True)
                
        # Initialize CodebaseManager without requiring a loaded repository
        codebase_manager = get_codebase_manager(workspace_dir=str(sandbox_root))
        
        # # Load the index if it exists
        # index_manager = codebase_manager.get_index_manager()
        # if os.path.exists(index_manager.faiss_index_file):
        #     index_manager.load_index()
        #     logger.info(f"Existing index loaded from: {index_manager.faiss_index_file}")
        # else:
        #     logger.info("No existing index found. Will create new index when needed.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.post("/api/codebase/import")
async def import_codebase(request: ImportRequest):
    # Get or create the manager instance
    codebase_manager = get_codebase_manager(workspace_dir="sandbox")
    logger.info(f"Before import - manager state: {vars(codebase_manager)}")
    
    # Import the codebase
    success = codebase_manager.import_codebase(request.source)
    logger.info(f"After import - manager state: {vars(codebase_manager)}")
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to import codebase")
        
    logger.info(f"Successfully imported codebase to: {codebase_manager.repo_dir}")
    
    # Verify the index manager was created
    logger.info(f"Index manager after import: {codebase_manager.index_manager}")
    
    analysis = codebase_manager.analyze_codebase()
    
    if not analysis:
        raise HTTPException(status_code=500, detail="Analysis failed")
        
    # Ensure all required fields exist
    analysis.setdefault('files', {})
    analysis.setdefault('code', {})
    analysis.setdefault('git', {})
    analysis.setdefault('llm_processing', {
        'total_characters': 0,
        'can_process_with_gpt4': False,
        'concatenated_code': ''
    })
    analysis.setdefault('enhanced_stats', {
        'estimated_read_time': 'N/A'
    })
    
    logger.info(f"Returning analysis: {analysis}")
    
    return JSONResponse(content={
        "status": "success",
        "data": analysis
    })

    # except Exception as e:
    #     logger.error(f"Error analyzing codebase: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))
    # finally:
    #     # Only cleanup if not keeping workspace
    #     if not request.keep_workspace:
    #         manager.cleanup(keep_workspace=False)

    # return {"status": "success", "message": "Codebase imported and indexed successfully"}