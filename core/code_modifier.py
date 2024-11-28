"""
modifier.py

This module contains the Modifier class, which is responsible for modifying existing code using LLM.

Classes:
    None

Functions:
    modify_code: Modifies existing Python code using OpenAI's GPT model based on user requests.

Dependencies:
    - openai: For interacting with OpenAI's API
    - dotenv: For loading environment variables
    - utils.sanitizer: For cleaning code output
    - utils.config: For logging configuration
"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: core/modifier.py

import traceback
from openai import OpenAI
from dotenv import load_dotenv
from utils.sanitizer import clean_code
import os
from utils.config import logger, client
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from utils.search import search_code
import ast
from custom_mapper2 import get_codebase_metadata, condense_codebase_metadata
from core.codebase_manager import CodebaseManager
from core.code_mapper import CodeMapper
from fastapi import HTTPException
from collections import defaultdict
from pathlib import Path
import json
from core.codebase_manager import get_codebase_manager
from utils.config import OPENAI_CHAT_MODEL

# OPENAI_CHAT_MODEL = "gpt-4o-mini"

@dataclass
class ModificationContext:
    original_code: str
    request: str
    filepath: Optional[str]
    related_files: List[Dict]
    # dependencies: List[Dict]
    evaluation_result: Optional[str] = None
    metadata: Optional[Dict] = None
    rag_query: Optional[str] = None

# class CodeModifier:

#     def __init__(self, code_mapper: CodeMapper):
#         self.context = None
#         self.max_iterations = 5
#         self.codebase_metadata = None
#         self.condensed_codebase_metadata = None
#         self.code_mapper = code_mapper

#     def _generate_rag_query(self, request: str, directory_structure: Dict) -> str:
#         """Generate an optimized RAG query based on the user request and codebase metadata."""
#         prompt = f"""Given a code modification request and the codebase structure, generate an optimized RAG query.

#         The query should help identify the most relevant code sections and dependencies. The query should be concise and specific. It will be used in a vector search to find relevant context in the codebase.

#         Modification request:
#         {request}

#         Directory structure and metadata:
#         {directory_structure}

#         Generate a search query that will:
#         1. Identify core functionality related to the request
#         2. Find similar patterns in the codebase
#         3. Locate relevant utility functions and dependencies
#         4. Focus on the specific domain of the modification

#         Return only the optimized search query without any explanation or additional text. Do not include ```python or ``` in the query. Do not include AND OR OR NOT in the query. Respond with a complete query, beginning with ```plaintext and ending with ```.
#         """

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are an expert at generating precise search queries for codebases."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.3
#             )
            
#             rag_query = response.choices[0].message.content.strip()
#             cleaned_rag_query = rag_query.replace("```plaintext", "").replace("```", "")
#             logger.info(f"Generated RAG query: {cleaned_rag_query}")
#             return cleaned_rag_query
        

#         except Exception as e:
#             logger.error(f"Error generating RAG query: {e}")
#             logger.info(f"Falling back to original request: {request}")
#             logger.info(f"Traceback: {traceback.format_exc()}")
#             return request  # Fallback to original request if query generation fails
        

#     def _gather_initial_context(self, original_code: str, request: str, filepath: str = None) -> ModificationContext:
#         """Gather initial context including related files and dependencies."""
#         logger.info("Gathering initial modification context")
        
#         # Map directory structure and cross-references
#         self.codebase_metadata = get_codebase_metadata()

#         # Condense metadata for relevant context
#         self.condensed_codebase_metadata = condense_codebase_metadata(self.codebase_metadata)
        
#         # Generate optimized RAG query
#         self.rag_query = self._generate_rag_query(request, self.condensed_codebase_metadata)
        
#         # Get directly related files through RAG using the optimized query
#         related_files = search_code(self.rag_query, k=3)
        
#         logger.info(f"Related files: {related_files}")
        
#         # Find dependencies and imported files
#         # dependencies = self._analyze_dependencies(original_code, filepath)
        
#         return ModificationContext(
#             original_code=original_code,
#             request=request,
#             filepath=filepath,
#             related_files=related_files,
#             # dependencies=dependencies,
#             metadata=self.condensed_codebase_metadata,
#             rag_query=self.rag_query
#         )

    
#     def _extract_imports(self, node: ast.AST) -> List[str]:
#         """Extract import names from AST node."""
#         if isinstance(node, ast.Import):
#             return [name.name for name in node.names]
#         elif isinstance(node, ast.ImportFrom):
#             return [f"{node.module}.{name.name}" for name in node.names]
#         return []

#     def _evaluate_modification(self, original: str, modified: str, request: str) -> bool:
#         """Evaluate if the modification meets the requirements."""
#         eval_prompt = f"""Evaluate if the code modification meets the requirements.
        
#         Original code:
#         ```python
#         {original}
#         ```
        
#         Modified code:
#         ```python
#         {modified}
#         ```
        
#         Modification request:
#         {request}
        
#         Evaluate and respond with either:
#         - "PASS" if the modification satisfies the request and maintains code quality
#         - "FAIL: <reason>" if the modification needs improvement
#         """
        
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a Python code review expert. Be strict about code quality and requirement satisfaction."},
#                     {"role": "user", "content": eval_prompt}
#                 ],
#                 temperature=0.6
#             )
            
#             result = response.choices[0].message.content.strip()
#             return result.startswith("PASS"), result
            
#         except Exception as e:
#             logger.error(f"Error in modification evaluation: {e}")
#             return False, f"FAIL: Evaluation error - {str(e)}"



#     def _build_context_prompt(self, instruction: str, metadata: str, modifications: list, current_loop: int) -> str:
#         """Build the context prompt for the LLM."""
#         if current_loop == 0:
#             return f"""
#             Codebase Metadata:
#             {metadata}

#             User Instruction:
#             {instruction}

#             Respond with a JSON array containing file modifications. Each object should have:
#             - filepath: string
#             - lines: array of integers
#             """
#         else:
#             return f"""
#             Codebase Metadata:
#             {metadata}

#             User Instruction:
#             {instruction}

#             Previous Modifications:
#             {json.dumps(modifications, indent=2)}

#             Assess if the modifications fully address the user's intent.
#             If not, propose additional changes needed.
            
#             Respond with a JSON array containing additional needed modifications.
#             """



#     def modify_code(self, instruction: str, current_code: str, filepath: str, context: ModificationContext) -> str:
#         """Modify code based on user request and context."""
#         if not self.code_mapper.repo_dir:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No repository has been imported. Please import a codebase first."
#             )
        
#         # Get the repo path from code_mapper and resolve it to absolute path
#         repo_path = Path(self.code_mapper.repo_dir).resolve()
#         logger.info(f"Using repository path: {repo_path}")
        
#         ignore_files = ["__pycache__", ".venv", ".git", ".github", "logs", "sandbox", "generated_code"]
#         codebase_metadata = get_codebase_metadata(repo_path, ignore_files)
#         condensed_codebase_metadata = condense_codebase_metadata(codebase_metadata)

#         system_prompt = """
#         You are an expert coding assistant. Using the provided code context and file content, 
#         create a detailed plan for retrieving the relevant code context. Focus on the specific file 
#         and location mentioned in the context. Provide paths relative to the repository root.
#         """

#         max_reflection_loops = 5
#         current_loop = 0
#         all_modifications = []
#         file_contents = {}
#         modified_files = set()

#         while current_loop < max_reflection_loops:
#             context_prompt = self._build_context_prompt(
#                 instruction, 
#                 condensed_codebase_metadata, 
#                 all_modifications, 
#                 current_loop
#             )
            
#             context_response = self._get_llm_response(system_prompt, context_prompt)
#             context_data = self._parse_json_response(context_response)
            
#             if not context_data:
#                 break

#             new_modifications = self._process_file_modifications(
#                 context_data,
#                 repo_path,
#                 instruction,
#                 all_modifications,
#                 file_contents
#             )
            
#             if not new_modifications:
#                 break

#             all_modifications.extend(new_modifications)
#             modified_files.update(mod["filepath"] for mod in new_modifications)
#             current_loop += 1

#         # Prepare the final response
#         response = {
#             "files": {},
#             "evaluation_result": "",
#             "reflection_loops": current_loop,
#             "files_modified": list(modified_files)
#         }

#         # Include the final state of each modified file
#         for rel_filepath in modified_files:
#             try:
#                 # Construct absolute path by joining with repo_path directly
#                 abs_filepath = repo_path / rel_filepath.lstrip('/')
#                 logger.info(f"Reading modified file from: {abs_filepath}")
                
#                 with open(abs_filepath, 'r') as f:
#                     response["files"][rel_filepath] = f.read()
#             except Exception as e:
#                 logger.error(f"Error reading modified file {rel_filepath}: {e}")
#                 response["files"][rel_filepath] = f"Error reading file: {str(e)}"

#         return response

#     def _normalize_path(self, filepath: str) -> str:
#         """Normalize a filepath to be relative to repo root."""
#         # Remove any leading/trailing whitespace and slashes
#         filepath = filepath.strip().strip('/')
        
#         # Convert Windows paths to Unix-style
#         filepath = filepath.replace('\\', '/')
        
#         # Split the path and remove any empty parts
#         parts = [p for p in filepath.split('/') if p and p != '.']
        
#         # If the path starts with sandbox/CodeRAG, remove it
#         if len(parts) >= 2:
#             if parts[0] == 'sandbox' and parts[1] == 'CodeRAG':
#                 parts = parts[2:]
        
#         return '/'.join(parts)

#     def _process_file_modifications(self, context_data: list, repo_path: Path, 
#                                 instruction: str, previous_modifications: list,
#                                 file_contents: dict) -> list:
#         """Process and apply file modifications."""
#         modifications = []
        
#         for item in context_data:
#             try:
#                 # Get relative path
#                 rel_filepath = self._normalize_path(item.get("filepath", ""))
#                 if not rel_filepath:
#                     continue
                
#                 # Create absolute path by joining with repo_path
#                 abs_filepath = repo_path / rel_filepath
#                 logger.info(f"Processing file: {abs_filepath} (relative: {rel_filepath})")
                
#                 # Ensure directory exists
#                 abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                
#                 # Cache handling
#                 if not abs_filepath.exists():
#                     file_contents[rel_filepath] = []
#                 elif rel_filepath not in file_contents:
#                     with open(abs_filepath, 'r') as f:
#                         file_contents[rel_filepath] = f.readlines()
                
#                 # Get modifications from LLM
#                 mod_prompt = self._build_modification_prompt(
#                     rel_filepath,
#                     item.get("lines", []), 
#                     file_contents[rel_filepath],
#                     instruction, 
#                     previous_modifications
#                 )
                
#                 mod_response = self._get_llm_response(
#                     "You are a code modification expert. Provide specific, line-by-line changes.",
#                     mod_prompt
#                 )
                
#                 new_mods = self._parse_json_response(mod_response)
#                 if not new_mods:
#                     continue
                    
#                 # Update filepath in modifications to use relative path
#                 for mod in new_mods:
#                     mod["filepath"] = rel_filepath
                    
#                 # Apply modifications
#                 modified_contents = self._apply_modifications(
#                     file_contents[rel_filepath],
#                     new_mods
#                 )
                
#                 # Update file contents
#                 file_contents[rel_filepath] = modified_contents
                
#                 # Write changes to file
#                 with open(abs_filepath, 'w') as f:
#                     f.writelines(modified_contents)
                
#                 modifications.extend(new_mods)
                
#             except Exception as e:
#                 logger.error(f"Error processing file {item.get('filepath')}: {e}")
#                 continue
        
#         return modifications

#     def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
#         """Get response from LLM."""
#         try:
#             response = client.chat.completions.create(
#                 model=OPENAI_CHAT_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.3,
#                 max_tokens=4000
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             logger.error(f"Error getting LLM response: {e}")
#             raise HTTPException(status_code=500, detail="Failed to get AI response")



#     def _build_modification_prompt(self, filepath: Path, lines: list, current_contents: list, 
#                                 instruction: str, previous_modifications: list) -> str:
#         """
#         Build a prompt for the LLM to get specific file modifications.
        
#         Args:
#             filepath: Path object of the file to modify
#             lines: List of line numbers to consider
#             current_contents: Current contents of the file as a list of lines
#             instruction: Original user instruction
#             previous_modifications: List of previous modifications made
        
#         Returns:
#             str: Formatted prompt for the LLM
#         """
#         # Get the relevant code context
#         context_lines = []
#         for line_num in lines:
#             # Convert to 0-based index
#             idx = line_num - 1
#             if 0 <= idx < len(current_contents):
#                 # Add a few lines before and after for context
#                 start = max(0, idx - 2)
#                 end = min(len(current_contents), idx + 3)
                
#                 # Add line numbers and code
#                 for i in range(start, end):
#                     prefix = ">" if i == idx else " "
#                     context_lines.append(f"{prefix} {i+1}: {current_contents[i].rstrip()}")

#         # Get previous modifications for this file
#         file_previous_mods = [
#             mod for mod in previous_modifications 
#             if str(mod.get("filepath")) == str(filepath)
#         ]

#         # Build the prompt
#         prompt = f"""
#     File: {filepath}
#     Target Lines: {lines}

#     Current Code Context:
#     ```
#     {chr(10).join(context_lines)}
#     ```

#     User Instruction:
#     {instruction}

#     Previous Modifications to this File:
#     {json.dumps(file_previous_mods, indent=2) if file_previous_mods else "None"}

#     Please provide specific modifications needed for the target lines.
#     Return a JSON array of modifications, where each modification object includes:
#     {{
#         "filepath": "{str(filepath)}",
#         "line": <line_number>,
#         "operation": "modify"|"insert"|"delete",
#         "modified_code": <new_code>,
#         "notes": <explanation>
#     }}

#     For insertions, the line number indicates where to insert the new code.
#     For modifications, provide the complete new line of code.
#     For deletions, simply specify the line to delete.

#     Important:
#     - Keep modifications minimal and focused
#     - Preserve indentation and coding style
#     - Ensure syntactic correctness
#     - Consider the impact on surrounding code
#     - Include clear notes explaining each change
#     """

#         return prompt

#     def _parse_json_response(self, content: str) -> list:
#         """Parse JSON response from LLM."""
#         try:
#             json_str = content.strip()
#             if '```json' in json_str:
#                 json_str = json_str.split('```json')[1].split('```')[0].strip()
                
#             # If response is a single object, wrap it in a list
#             parsed = json.loads(json_str)
#             if isinstance(parsed, dict):
#                 parsed = [parsed]
                
#             return parsed
            
#         except Exception as e:
#             logger.error(f"Error parsing JSON response: {e}")
#             return []

#     # Format the response in a more editor-friendly way
#     def format_code_changes(self, modifications):
#         formatted_changes = []
#         for mod in modifications:
#             change = (
#                 f"// File: {mod['filepath']}\n"
#                 f"// Line {mod['line']}\n"
#                 f"// Original:\n{mod['original']}\n"
#                 f"// Modified:\n{mod['modified']}\n"
#             )
#             formatted_changes.append(change)
        
#         return "\n".join(formatted_changes)

#     def extract_code_from_lines(self, full_code: str, lines: List[int]) -> str:
#         """Extract code from a given file content based on specified lines."""
#         return "\n".join([full_code.split("\n")[i-1] for i in lines])


#     def _trim_context(self, context_parts: List[str], max_tokens: int) -> List[str]:
#         """Trim context parts to fit within the token limit by re-querying the RAG."""
#         logger.info("Trimming context to fit within token limits")
        
#         # Reconstruct a query from the modification request and current context
#         reconstructed_query = self._reconstruct_query()
        
#         # Re-query the RAG for the most relevant context
#         refined_results = search_code(reconstructed_query, k=5)
        
#         # Build a new context from the refined results
#         refined_context = []
#         current_tokens = 0
        
#         for result in refined_results:
#             part = f"\nFile: {result['filepath']}\n```python\n{result['content']}\n```"
#             part_tokens = len(part.split())
            
#             if current_tokens + part_tokens <= max_tokens:
#                 refined_context.append(part)
#                 current_tokens += part_tokens
#             else:
#                 logger.info(f"Skipping context part due to token limit: {part[:50]}...")
#                 break
        
#         return refined_context

#     def _reconstruct_query(self) -> str:
#         """Reconstruct a query from the modification request and current context."""
#         # Use the modification request as the base
#         query = self.context.request
        
#         # Optionally, add key insights from the current context
#         if self.context.related_files:
#             query += " " + " ".join(file['filepath'] for file in self.context.related_files)
        
#         # if self.context.dependencies:
#         #     query += " " + " ".join(dep['filepath'] for dep in self.context.dependencies)
        
#         logger.info(f"Reconstructed query for RAG: {query}")
#         return query


class CodeModifier:
    def __init__(self, code_mapper: CodeMapper):
        self.code_mapper = code_mapper
        self.context = None
        self.max_iterations = 3
        self.codebase_metadata = None
        self.condensed_codebase_metadata = None
        
    def modify_code(self, instruction: str, current_code: str, filepath: str, context: ModificationContext) -> str:
        """Modify code based on user request and context."""
        if not self.code_mapper.repo_dir:
            raise HTTPException(
                status_code=400,
                detail="No repository has been imported. Please import a codebase first."
            )
        
        repo_path = Path(self.code_mapper.repo_dir).resolve()
        logger.info(f"Using repository path: {repo_path}")
        
        # Initialize metadata
        ignore_files = ["__pycache__", ".venv", ".git", ".github", "logs", "sandbox", "generated_code"]
        self.codebase_metadata = get_codebase_metadata(repo_path, ignore_files)
        self.condensed_codebase_metadata = condense_codebase_metadata(self.codebase_metadata)

        # Phase 1: Planning - Determine what modifications to make
        planned_modifications = self._plan_modifications(instruction, repo_path)
        
        # Phase 2: Validation - Verify the planned modifications
        validated_modifications = self._validate_modifications(planned_modifications)
        
        # Phase 3: Execution - Apply the validated modifications
        results = self._execute_modifications(validated_modifications, repo_path)
        
        return results

    def _apply_modifications(self, lines: list, modifications: list) -> list:
        """Apply modifications to file contents."""
        # Sort modifications in reverse order to handle line numbers correctly
        
        
        if isinstance(lines, str):
            lines = lines.splitlines(keepends=True)
        elif not isinstance(lines, list):
            lines = list(lines)
        
        modifications.sort(key=lambda x: x["line"], reverse=True)
        
        new_lines = lines.copy()
        
        for mod in modifications:
            line_num = mod["line"] - 1  # Convert to 0-based index
            operation = mod["operation"]
            modified_code = mod["modified_code"]
            
            if operation == "insert":
                new_lines.insert(line_num, modified_code + "\n")
            elif operation == "delete" and 0 <= line_num < len(new_lines):
                new_lines.pop(line_num)
            elif operation == "modify" and 0 <= line_num < len(new_lines):
                # For modifications, ensure we're replacing the exact line specified
                # and preserve any existing line ending
                original_line_ending = "\n" if new_lines[line_num].endswith("\n") else ""
                new_lines[line_num] = modified_code + original_line_ending        
        return new_lines
    
    def _plan_modifications(self, instruction: str, repo_path: Path) -> list:
        """Plan what modifications should be made without actually making them."""
        system_prompt = """You are a code modification planner. Your task is to plan specific code changes 
        without executing them. Focus on determining exactly what changes should be made where."""
        
        # Extract modification count from instruction
        # modification_count = self._extract_modification_count(instruction)
        # logger.info(f"Planning {modification_count} modifications")
        
        # Build initial planning prompt
        planning_prompt = self._build_planning_prompt(
            instruction,
            self.condensed_codebase_metadata
        )
        
        # Iteratively refine the plan
        current_plan = []
        max_planning_iterations = 3
        
        for iteration in range(max_planning_iterations):
            logger.info(f"Planning iteration {iteration + 1}")
            
            # Get next iteration of the plan
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": planning_prompt + self._format_current_plan(current_plan)}
                ],
                temperature=0.7  # Allow some creativity in planning
            )
            
            # Parse the response into a structured plan
            new_plan = self._parse_planning_response(response.choices[0].message.content)
            
            # Check if the plan has stabilized
            if self._plans_are_equivalent(current_plan, new_plan):
                logger.info("Plan has stabilized")
                break
                
            current_plan = new_plan
            
            # Update prompt for next iteration
            planning_prompt = self._build_refinement_prompt(instruction, current_plan)
        
        return current_plan

    def _validate_modifications(self, planned_modifications: list) -> list:
        """Validate the planned modifications before execution."""
        validated_mods = []
        
        for mod in planned_modifications:
            try:
                # Validate basic structure
                required_fields = ["filepath", "line", "operation", "modified_code"]
                if not all(field in mod for field in required_fields):
                    logger.warning(f"Skipping modification missing required fields: {mod}")
                    continue
                
                # Validate filepath exists
                filepath = Path(self.code_mapper.repo_dir) / mod["filepath"]
                if not filepath.exists():
                    logger.warning(f"Skipping modification for non-existent file: {filepath}")
                    continue
                
                # Validate line number
                with open(filepath, 'r') as f:
                    file_lines = f.readlines()
                if not (0 <= mod["line"] - 1 < len(file_lines)):
                    logger.warning(f"Skipping modification with invalid line number: {mod}")
                    continue
                
                # Validate operation type
                if mod["operation"] not in ["insert", "modify", "delete"]:
                    logger.warning(f"Skipping modification with invalid operation: {mod}")
                    continue
                
                validated_mods.append(mod)
                
            except Exception as e:
                logger.error(f"Error validating modification: {e}")
                continue
        
        return validated_mods

    def _execute_modifications(self, validated_modifications: list, repo_path: Path) -> dict:
        """Execute the validated modifications."""
        modified_files = {}
        file_contents = {}
        
        # Group modifications by file
        mods_by_file = {}
        for mod in validated_modifications:
            filepath = mod["filepath"]
            if filepath not in mods_by_file:
                mods_by_file[filepath] = []
            mods_by_file[filepath].append(mod)
        
        # Apply modifications file by file
        for filepath, mods in mods_by_file.items():
            try:
                abs_path = repo_path / filepath
                
                # Read file content
                with open(abs_path, 'r') as f:
                    lines = f.readlines()
                
                # Sort modifications in reverse order to maintain line numbers
                mods.sort(key=lambda x: x["line"], reverse=True)
                
                # Apply modifications
                modified_lines = self._apply_modifications(lines, mods)
                
                # Store modified content
                modified_files[filepath] = "".join(modified_lines)
                
                # Write changes to file
                with open(abs_path, 'w') as f:
                    f.writelines(modified_lines)
                
            except Exception as e:
                logger.error(f"Error applying modifications to {filepath}: {e}")
                continue
        
        return {
            "files": modified_files,
            "modifications": validated_modifications,
            "files_modified": list(modified_files.keys())
        }

    def _build_planning_prompt(self, instruction: str, metadata: dict) -> str:
        """Build the initial planning prompt."""
        return f"""
        Plan modifications to the codebase based on this instruction:
        {instruction}

        Available files and their structure:
        {json.dumps(metadata, indent=2)}

        Requirements:
        1. Each modification must specify:
           - filepath: relative path to the file
           - line: specific line number
           - operation: type of change (insert/modify/delete)
           - modified_code: the new code to add
           - notes: explanation of the change
        2. Use files that actually exist in the structure
        3. Choose appropriate locations for modifications
        4. Ensure modifications are consistent with the instruction

        Return the plan as a JSON array of modifications.
        Each modification should be complete and ready for validation.
        """

    def _build_refinement_prompt(self, instruction: str, current_plan: list) -> str:
        """Build a prompt for refining the current plan."""
        return f"""
        Review and refine this plan for modifications based on:
        {instruction}

        Current plan:
        {json.dumps(current_plan, indent=2)}

        Requirements:
        1. Ensure each modification is specific and complete
        2. Verify file paths and line numbers are appropriate
        3. Check that modifications fulfill the original instruction
        4. Look for potential improvements or better locations

        Return the refined plan as a JSON array of modifications.
        """

    # def _extract_modification_count(self, instruction: str) -> int:
    #     """Extract the number of modifications from the instruction."""
    #     import re
    #     match = re.search(r'(?:add|place|insert)\s+(\d+)', instruction.lower())
    #     return int(match.group(1)) if match else 1

    def _plans_are_equivalent(self, plan1: list, plan2: list) -> bool:
        """Check if two plans are functionally equivalent."""
        if len(plan1) != len(plan2):
            return False
            
        # Sort both plans by filepath and line number for comparison
        sorted_plan1 = sorted(plan1, key=lambda x: (x.get("filepath", ""), x.get("line", 0)))
        sorted_plan2 = sorted(plan2, key=lambda x: (x.get("filepath", ""), x.get("line", 0)))
        
        # Compare essential elements of each modification
        for mod1, mod2 in zip(sorted_plan1, sorted_plan2):
            if (mod1.get("filepath") != mod2.get("filepath") or
                mod1.get("line") != mod2.get("line") or
                mod1.get("operation") != mod2.get("operation") or
                mod1.get("modified_code") != mod2.get("modified_code")):
                return False
                
        return True

    def _format_current_plan(self, plan: list) -> str:
        """Format the current plan for inclusion in prompts."""
        return "\nCurrent plan:\n" + json.dumps(plan, indent=2) if plan else "\nNo current plan."

    def _parse_planning_response(self, response: str) -> list:
        """Parse the LLM's response into a structured plan."""
        try:
            # Extract JSON from response if wrapped in markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Parse JSON response
            plan = json.loads(response.strip())
            
            # Ensure result is a list
            if isinstance(plan, dict):
                plan = [plan]
                
            return plan
            
        except Exception as e:
            logger.error(f"Error parsing planning response: {e}")
            return []