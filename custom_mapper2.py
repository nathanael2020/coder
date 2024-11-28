import os
import json
from hashlib import sha256
from datetime import datetime
from utils.config import client
import ast
import traceback
import sys
from utils.config import logger
from pathlib import Path
from typing import Dict

# Constants
MODEL_NAME = "gpt-4o-mini"
OUTPUT_FILE = f"meta_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def hash_content(content):
    return sha256(content.encode("utf-8")).hexdigest()

total_tokens = 0

def call_llm(prompt, context=None, metadata=None):
    """
    Enhanced LLM caller with relevant context.
    """
    if metadata:
        relevant_metadata = extract_relevant_metadata(metadata, context)
        enhanced_prompt = f"""
Project Context:
{json.dumps(relevant_metadata, indent=2)}

Task:
{prompt}
"""
    else:
        enhanced_prompt = prompt

    # try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert software code reviewer."},
            {"role": "user", "content": enhanced_prompt}
        ],
        max_tokens=2000,
        temperature=0.7
    )
    print(response)
    print(response.usage)
    print(response.usage.total_tokens)

    global total_tokens
    total_tokens += response.usage.total_tokens

    print(f"Total tokens: {total_tokens}")
    # sys.exit()

    return response.choices[0].message.content
        
    # except Exception as e:
    #     logger.error(f"Error calling LLM: {e}")
    #     logger.error("Traceback: ", traceback.format_exc())
    #     return "Error generating response"


def extract_relevant_metadata(metadata, context):
    """
    Extract only the relevant parts of metadata based on context.
    """
    relevant = {"project_summary": metadata["project_summary"]}
    if context.get("file_path"):
        file_path = context["file_path"]
        relevant["current_file"] = metadata["index"]["files"].get(file_path)
    if context.get("function_name"):
        func_name = context["function_name"]
        relevant["current_function"] = metadata["index"]["functions"].get(func_name)
    return relevant


def condense_codebase_metadata(metadata):
    project_summary = {
        "name": metadata["project"]["name"],
        "description": metadata["project"]["description"],
        "entry_points": metadata["project"]["entry_points"],
        "dependencies": metadata["project"]["dependencies"],
    }

    index = {"files": {}, "functions": {}, "classes": {}}
    for file_name, file_data in metadata["files"].items():
        index["files"][file_name] = {"description": file_data["description"]}
        for func_name, func_data in file_data.get("functions", {}).items():
            index["functions"][func_name] = {
                "file": file_name,
                "lines": func_data["line_range"],
                "description": func_data["description"],
            }
        for class_name, class_data in file_data.get("classes", {}).items():
            index["classes"][class_name] = {
                "file": file_name,
                "lines": class_data["line_range"],
                "description": class_data["description"],
            }

    return {"project_summary": project_summary, "index": index}


class LLMReflector:
    def __init__(self, metadata, condensed_meta_log):
        self.metadata = metadata
        self.condensed_meta_log = condensed_meta_log
        self.logs = []
        self.context = {}
        self.recursion_depth = 0

    def reflect(self, objective):
        reflection_prompt = f"""
Objective: {objective}

Current Context:
{json.dumps(self.context, indent=2)}

Condensed Meta Log:
{json.dumps(self.condensed_meta_log, indent=2)}

Do you have enough information to answer the objective? 
- If yes, provide the answer.
- If no, specify what additional information you need.
"""
        
        print(reflection_prompt)
        response = call_llm(reflection_prompt)
        print(response)
        self.logs.append({"context": self.context, "response": response})
        return self.process_response(response)

    def process_response(self, response):
        if "I have enough information" in response:
            return {"status": "complete", "answer": response}
        else:
            requests = self.parse_requests(response)
            return {"status": "incomplete", "requests": requests}

    def parse_requests(self, response):
        if "Please provide" in response:
            return [req.strip() for req in response.split("Please provide")[1:]]
        return []

    def fetch_information(self, requests):
        for req in requests:
            if "entry points" in req:
                self.context["entry_points"] = self.condensed_meta_log["project_summary"]["entry_points"]
            elif "dependencies" in req:
                self.context["dependencies"] = self.condensed_meta_log["project_summary"]["dependencies"]

    def run(self, objective):
        self.context = {
            "project_name": self.condensed_meta_log["project_summary"]["name"],
            "project_description": self.condensed_meta_log["project_summary"]["description"],
        }

        while self.recursion_depth < 10:
            result = self.reflect(objective)
            if result["status"] == "complete":
                return result["answer"]
            self.fetch_information(result["requests"])
            self.recursion_depth += 1
        return "Failed to complete within recursion limits."



def get_codebase_metadata(root_dir, ignore_files=None):
    """Map directory structure with cross-referencing."""
    project_metadata = {
        "project": {
            "name": os.path.basename(os.path.abspath(root_dir)),
            "description": "Metadata for the project.",
            "entry_points": [],
            "themes": [],
            "dependencies": [],
            "modules": {},
        },
        "files": {},
        "cross_references": {
            "call_graph": {},
            "dependency_graph": {},
        },
    }

    ignore_dirs = {'.venv', '.git', '.github', 'logs', '__pycache__', 'sandbox', 'generated_code'}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        # filenames = [f for f in filenames if f not in ignore_files]

        for file in filenames:
            if file.endswith('.py'):
                file_path = os.path.join(dirpath, file)
                relative_file_path = os.path.relpath(file_path, root_dir)
                file_metadata = extract_metadata_from_file(file_path)

                # Build cross-references
                for func, meta in file_metadata.get("functions", {}).items():
                    project_metadata["cross_references"]["call_graph"].setdefault(func, []).extend(meta["dependencies"])
                for dep in file_metadata.get("imports", []):
                    project_metadata["cross_references"]["dependency_graph"].setdefault(dep, []).append(relative_file_path)

                project_metadata["files"][relative_file_path] = file_metadata

                # Aggregate project-level metadata
                project_metadata["project"]["themes"].extend(file_metadata.get("themes", []))
                project_metadata["project"]["dependencies"].extend(file_metadata.get("imports", []))
                project_metadata["project"]["entry_points"].extend(file_metadata.get("entry_points", []))

    # Remove duplicates
    project_metadata["project"]["themes"] = list(set(project_metadata["project"]["themes"]))
    project_metadata["project"]["dependencies"] = list(set(project_metadata["project"]["dependencies"]))
    project_metadata["project"]["entry_points"] = list(set(project_metadata["project"]["entry_points"]))

    # Generate project-level summary
    project_metadata["project"]["llm_summary"] = "LLM response placeholder"  # Replace with actual LLM call

    return project_metadata


# def get_codebase_metadata(repo_path: Path = None) -> Dict:
#     """Get metadata about the codebase structure."""
#     if repo_path is None:
#         manager = get_codebase_manager()
#         repo_path = manager.get_repo_path()
    
#     metadata = {}
#     try:
#         # Walk through the repository directory instead of current directory
#         for root, dirs, files in os.walk(repo_path):
#             # Skip hidden directories and files
#             dirs[:] = [d for d in dirs if not d.startswith('.')]
#             files = [f for f in files if not f.startswith('.')]
            
#             # Get relative path from repo root
#             rel_path = os.path.relpath(root, repo_path)
#             if rel_path == '.':
#                 rel_path = ''
                
#             # Process files...
#             # Rest of the existing function logic

#     except Exception as e:
#         metadata["error"] = str(e)

#     return metadata



def extract_dependencies(node):
    """Extract dependencies (function calls) from a node."""
    dependencies = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                dependencies.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                dependencies.add(child.func.attr)
    return list(dependencies)


def extract_metadata_from_file(file_path):
    """Extract metadata for a single file."""
    metadata = {
        "description": "",
        "entry_points": [],
        "themes": [],
        "imports": [],
        "classes": {},
        "functions": {},
        "has_main": False  # New field to track main block
    }
    try:
        with open(file_path, "r") as file:
            content = file.read()
            tree = ast.parse(content)

            # Extract module-level docstring
            metadata["description"] = ast.get_docstring(tree) or ""

            # Check for if __name__ == "__main__": block
            for node in tree.body:
                if (isinstance(node, ast.If) and 
                    isinstance(node.test, ast.Compare) and 
                    isinstance(node.test.left, ast.Name) and 
                    node.test.left.id == "__name__" and 
                    isinstance(node.test.comparators[0], ast.Constant) and 
                    node.test.comparators[0].value == "__main__"):
                    metadata["has_main"] = True
                    # Extract function calls within main block
                    for n in ast.walk(node):
                        if isinstance(n, ast.Call):
                            if isinstance(n.func, ast.Name):
                                metadata["entry_points"].append(n.func.id)
                            elif isinstance(n.func, ast.Attribute):
                                metadata["entry_points"].append(n.func.attr)

            # Extract imports
            metadata["imports"] = [
                node.names[0].name for node in tree.body if isinstance(node, ast.Import)
            ] + [
                node.module for node in tree.body if isinstance(node, ast.ImportFrom)
            ]

            # Look for main() function definition
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "main":
                    metadata["has_main"] = True
                    metadata["entry_points"].append("main")

                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_doc = ast.get_docstring(node) or ""
                    class_metadata = {
                        "line_range": [node.lineno, getattr(node, 'end_lineno', node.lineno)],
                        "description": class_doc,
                        "methods": {}
                    }

                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            method_doc = ast.get_docstring(item) or ""
                            method_metadata = {
                                "line_range": [item.lineno, getattr(item, 'end_lineno', item.lineno)],
                                "description": method_doc,
                                "parameters": [arg.arg for arg in item.args.args],
                                "dependencies": extract_dependencies(item),
                            }
                            class_metadata["methods"][method_name] = method_metadata

                    metadata["classes"][class_name] = class_metadata

                elif isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    function_doc = ast.get_docstring(node) or ""
                    function_metadata = {
                        "line_range": [node.lineno, getattr(node, 'end_lineno', node.lineno)],
                        "description": function_doc,
                        "parameters": [arg.arg for arg in node.args.args],
                        "dependencies": extract_dependencies(node),
                    }
                    metadata["functions"][function_name] = function_metadata

                    # Identify entry points (e.g., FastAPI endpoints)
                    if hasattr(node, 'decorator_list'):
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Attribute) and decorator.attr in {'get', 'post', 'put', 'delete'}:
                                metadata["entry_points"].append(function_name)

            # Generate themes and summary using LLM
            # llm_input = f"""
            # Analyze the following metadata and docstrings from the file '{os.path.basename(file_path)}':
            
            # Module Description:
            # {metadata['description']}
            
            # Classes:
            # {json.dumps(metadata['classes'], indent=2)}
            
            # Functions:
            # {json.dumps(metadata['functions'], indent=2)}
            
            # Provide a summary, themes, and key insights.
            # """

            # llm_response = call_llm(llm_input)

            # # llm_response = "LLM response"
            # metadata["llm_summary"] = llm_response

    except Exception as e:
        metadata["error"] = str(e)

    return metadata




def iterative_codebase_description(metadata, condensed_meta_log):
    reflector = LLMReflector(metadata, condensed_meta_log)
    objective = (
        "Create a detailed plan to implement to LLM modification according to user's request, including understanding the codebase structure, understanding the user's intent, creating new files, rewriting existing files, and soliciting user's confirmation."
    )
    result = reflector.run(objective)
    print("\nLogs of Reflections and Actions:")
    for log in reflector.logs:
        print(json.dumps(log, indent=2))
    return result



if __name__ == "__main__":
    ignore_files = {'.env', 'meta_log.json'}
    metadata = get_codebase_metadata(ignore_files=ignore_files)

    with open(f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    condensed_metadata = condense_codebase_metadata(metadata)

    with open(f"condensed_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(condensed_metadata, f, indent=2)

    description = iterative_codebase_description(metadata, condensed_metadata)
    with open(f"codebase_description_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(description)
