import os
import openai
import json
from hashlib import sha256
from datetime import datetime
from utils.config import client
import os
import ast
import json
from hashlib import sha256

# Constants
CHUNK_SIZE = 10  # Approximate number of lines per chunk
LOG_FILE = "meta_log.json"
# MODEL_NAME = "gpt-3.5-turbo"  # Choose the desired model
MODEL_NAME = "gpt-4o-mini"  # Choose the desired model

import ast

def chunk_file_syntax_aware(file_path):
    """
    Parse a Python file and chunk it by functions and classes.
    
    Performs syntax-aware chunking of Python source code by identifying function and class
    definitions using the AST parser. Each chunk contains metadata about the code segment.

    Args:
        file_path (str): Path to the Python file to be chunked

    Returns:
        List[Dict]: A list of chunks where each chunk contains:
            - id: Unique hash of the chunk content
            - type: Either "class" or "function"
            - name: Name of the class or function
            - content: Actual source code
            - start_line: Starting line number
            - end_line: Ending line number

    Raises:
        ValueError: If there's a syntax error in the source file
    """
    with open(file_path, "r") as file:
        source_code = file.read()

    try:
        # Parse the source code into an AST
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in file {file_path}: {e}")

    chunks = []
    for node in ast.walk(tree):
        # Only process function and class definitions
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = max(getattr(n, 'lineno', start_line) for n in ast.walk(node))
            chunk_content = '\n'.join(source_code.splitlines()[start_line - 1:end_line])
            
            # Create a chunk dictionary
            chunks.append({
                "id": f"{hash_content(chunk_content)}",
                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                "name": node.name,
                "content": chunk_content,
                "start_line": start_line,
                "end_line": end_line
            })

    return chunks

OUTPUT_FILE = f"meta_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def hash_content(content):
    """
    Generate a unique hash for a chunk of content using SHA-256.

    Args:
        content (str): The content to be hashed

    Returns:
        str: Hexadecimal string representation of the SHA-256 hash
    """
    return sha256(content.encode("utf-8")).hexdigest()



def call_llm(prompt, context=None, metadata=None):
    """
    Enhanced LLM caller that includes relevant context from metadata.
    
    Args:
        prompt (str): The main prompt for the LLM
        context (dict, optional): Specific context needed for this call
        metadata (dict, optional): Condensed project metadata
    """
    if metadata:
        # Only include relevant sections of metadata based on the context
        relevant_metadata = extract_relevant_metadata(metadata, context)
        
        enhanced_prompt = f"""
Project Context:
{json.dumps(relevant_metadata, indent=2)}

Task:
{prompt}
"""
    else:
        enhanced_prompt = prompt

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert software code reviewer."},
                {"role": "user", "content": enhanced_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Error generating response"

def extract_relevant_metadata(metadata, context):
    """
    Extract only the relevant parts of metadata based on context.
    
    Args:
        metadata (dict): Condensed project metadata
        context (dict): Context specifying what metadata is needed
    """
    relevant = {"project_summary": metadata["project_summary"]}
    
    if context.get("file_path"):
        file_path = context["file_path"]
        relevant["current_file"] = metadata["index"]["files"].get(file_path)
        
    if context.get("function_name"):
        func_name = context["function_name"]
        relevant["current_function"] = metadata["index"]["functions"].get(func_name)
        # Include related functions based on call graph
        relevant["related_functions"] = {
            name: metadata["index"]["functions"].get(name)
            for name in metadata["cross_references"]["call_graph"].get(func_name, [])
        }
        
    return relevant

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
            llm_input = f"""
            Analyze the following metadata and docstrings from the file '{os.path.basename(file_path)}':
            
            Module Description:
            {metadata['description']}
            
            Classes:
            {json.dumps(metadata['classes'], indent=2)}
            
            Functions:
            {json.dumps(metadata['functions'], indent=2)}
            
            Provide a summary, themes, and key insights.
            """
            # llm_response = call_llm(llm_input)

            llm_response = "LLM response"
            metadata["llm_summary"] = llm_response

    except Exception as e:
        metadata["error"] = str(e)

    return metadata

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


# def map_directory_structure(root_dir, ignore_files=None):
#     """Map directory structure and extract metadata."""
#     project_metadata = {
#         "project": {
#             "name": os.path.basename(os.path.abspath(root_dir)),
#             "description": "",
#             "entry_points": [],
#             "themes": [],
#             "dependencies": [],
#             "modules": {},
#         },
#         "files": {},
#     }

#     ignore_dirs = {'.venv', '.git', '.github', 'logs', '__pycache__', 'sandbox', 'generated_code'}

#     for dirpath, dirnames, filenames in os.walk(root_dir):

#         # Skip hidden directories and files
#         dirnames[:] = [d for d in dirnames if not d.startswith('.')]

#         dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

#         filenames = [f for f in filenames if not f.startswith('.')]

#         print(f"Processing directory: {dirpath}")
#         print(f"  Files: {filenames}")
#         # return

#         # Remove ignored files from filenames
#         filenames = [f for f in filenames if f not in ignore_files]

#         for file in filenames:

#             print(f"  Processing file: {file}")

#             if file.endswith('.py'):
#                 file_path = os.path.join(dirpath, file)
#                 relative_file_path = os.path.relpath(file_path, root_dir)

#                 print(f"    Extracting metadata from {file_path}")
#                 file_metadata = extract_metadata_from_file(file_path)
#                 print(f"    Metadata: {file_metadata}")
#                 project_metadata["files"][relative_file_path] = file_metadata

#                 # Collect project-level themes and dependencies
#                 project_metadata["project"]["themes"].extend(file_metadata.get("themes", []))
#                 project_metadata["project"]["dependencies"].extend(file_metadata.get("imports", []))
#                 project_metadata["project"]["entry_points"].extend(file_metadata.get("entry_points", []))

#     # Remove duplicates
#     project_metadata["project"]["themes"] = list(set(project_metadata["project"]["themes"]))
#     project_metadata["project"]["dependencies"] = list(set(project_metadata["project"]["dependencies"]))
#     project_metadata["project"]["entry_points"] = list(set(project_metadata["project"]["entry_points"]))

#     # Generate project-level summary using LLM
#     llm_input = f"""
#     Summarize the following project metadata:
    
#     Name: {project_metadata['project']['name']}
#     Description: {project_metadata['project']['description']}
#     Entry Points: {project_metadata['project']['entry_points']}
#     Themes: {project_metadata['project']['themes']}
#     Dependencies: {project_metadata['project']['dependencies']}
    
#     File-Level Metadata:
#     {json.dumps(project_metadata['files'], indent=2)}
#     """
#     # project_metadata["project"]["llm_summary"] = call_llm(llm_input)
#     project_metadata["project"]["llm_summary"] = "LLM response"
#     return project_metadata

def save_metadata_to_file(metadata, output_file=OUTPUT_FILE):
    """Save metadata to a JSON file."""
    with open(output_file, "w") as file:
        json.dump(metadata, file, indent=4)



def map_directory_structure_with_cross_references(root_dir, ignore_files=None):
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
        filenames = [f for f in filenames if f not in ignore_files]

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


def condense_metadata(metadata):
    # Build project summary
    project_summary = {
        "name": metadata["project"]["name"],
        "description": metadata["project"]["description"],
        "dependencies": metadata["project"]["dependencies"],
        "entry_points": metadata["project"]["entry_points"],
        "files_count": len(metadata["files"]),
        "modules_count": len({f.split("/")[0] for f in metadata["files"]})
    }

    # Index files, functions, and classes
    index = {"files": {}, "functions": {}, "classes": {}, "dependencies": {}}
    for file_name, file_data in metadata["files"].items():
        index["files"][file_name] = {
            "description": file_data["description"],
            "functions": file_data.get("functions", {}),
            "classes": file_data.get("classes", {}),
            "dependencies": file_data.get("imports", [])
        }
        for func_name, func_data in file_data.get("functions", {}).items():
            index["functions"][func_name] = {
                "file": file_name,
                "lines": func_data["line_range"],
                "description": func_data["description"],
                "dependencies": func_data.get("dependencies", [])
            }
        for class_name, class_data in file_data.get("classes", {}).items():
            index["classes"][class_name] = {
                "file": file_name,
                "lines": class_data["line_range"],
                "description": class_data["description"]
            }
        for dep in file_data.get("imports", []):
            if dep not in index["dependencies"]:
                index["dependencies"][dep] = []
            index["dependencies"][dep].append(file_name)

    # Build cross-references
    cross_references = {
        "dependencies": {dep: {"used_in": files} for dep, files in index["dependencies"].items()},
        "call_graph": metadata["cross_references"].get("call_graph", {})
    }

    return {
        "project_summary": project_summary,
        "index": index,
        "cross_references": cross_references
    }


def review_function(file_path, function_name, metadata):
    """Review a specific function with relevant context."""
    context = {
        "file_path": file_path,
        "function_name": function_name
    }
    
    prompt = f"Please review the function '{function_name}' from '{file_path}' and provide feedback on its implementation."
    
    return call_llm(prompt, context=context, metadata=metadata)

def discuss_codebase(metadata):
    """Generate a high-level discussion of the codebase."""
    context = {"scope": "project"}
    
    prompt = "Provide a high-level analysis of this codebase's architecture and design patterns."
    
    return call_llm(prompt, context=context, metadata=metadata)



def summarize_codebase(metadata):
    project_summary = metadata.get("project_summary", {})
    index = metadata.get("index", {})
    
    summary = {
        "name": project_summary.get("name"),
        "description": project_summary.get("description"),
        "entry_points": project_summary.get("entry_points", []),
        "files_count": project_summary.get("files_count", 0),
        "modules_count": project_summary.get("modules_count", 0),
        "dependencies": project_summary.get("dependencies", [])
    }
    
    return summary



def extract_major_elements(index):
    files = index.get("files", {})
    major_elements = {"entry_points": [], "key_classes": [], "key_functions": []}
    
    for file_name, file_data in files.items():
        if "main" in file_data.get("functions", {}):
            major_elements["entry_points"].append(file_name)
        
        major_elements["key_classes"].extend([
            {"name": class_name, **class_data}
            for class_name, class_data in file_data.get("classes", {}).items()
        ])
        
        major_elements["key_functions"].extend([
            {"name": func_name, **func_data}
            for func_name, func_data in file_data.get("functions", {}).items()
        ])
    
    return major_elements


def interpret_intent(summary, major_elements):
    intent = summary.get("description", "No description available.")
    
    # Infer purpose based on dependencies and major elements
    if "fastapi" in summary.get("dependencies", []):
        intent += " This appears to be a web API project, likely designed to handle HTTP requests."
    if "openai" in summary.get("dependencies", []):
        intent += " The project integrates OpenAI's GPT models, suggesting it includes generative AI features."
    if any("debug" in cls["name"].lower() for cls in major_elements["key_classes"]):
        intent += " Debugging and error-handling functionalities are emphasized."
    
    return intent



def analyze_codebase(index):
    feedback = {"strengths": [], "issues": []}
    
    # Assess strengths
    if "logging" in index.get("dependencies", {}):
        feedback["strengths"].append("Comprehensive logging functionality.")
    if "pytest" in index.get("dependencies", {}):
        feedback["strengths"].append("Includes unit tests for validation.")
    
    # Check for potential issues
    if not index.get("files"):
        feedback["issues"].append("The codebase lacks documentation on some files.")
    if not any(func["name"] == "main" for func in index.get("functions", {}).values()):
        feedback["issues"].append("No clearly defined entry point.")
    
    return feedback



def describe_codebase(metadata):
    # Step 1: Summarize the codebase
    summary = summarize_codebase(metadata)
    
    # Step 2: Identify major elements
    major_elements = extract_major_elements(metadata.get("index", {}))
    
    # Step 3: Interpret the intent
    intent = interpret_intent(summary, major_elements)
    
    # Step 4: Provide feedback
    feedback = analyze_codebase(metadata.get("index", {}))
    
    # Step 5: Combine into a narrative
    description = f"### Project: {summary['name']}\n"
    description += f"{summary['description']}\n\n"
    description += f"**Files:** {summary['files_count']} | **Modules:** {summary['modules_count']}\n"
    description += f"**Entry Points:** {', '.join(summary['entry_points'])}\n"
    description += f"**Dependencies:** {', '.join(summary['dependencies'])}\n\n"
    
    description += "### Major Elements\n"
    description += "#### Key Classes:\n"
    for cls in major_elements["key_classes"]:
        description += f"- {cls['name']} ({cls['file']}, lines {cls['lines'][0]}-{cls['lines'][1]}): {cls['description']}\n"
    description += "#### Key Functions:\n"
    for func in major_elements["key_functions"]:
        description += f"- {func['name']} ({func['file']}, lines {func['lines'][0]}-{func['lines'][1]}): {func['description']}\n\n"
    
    description += f"### Intent\n{intent}\n\n"
    
    description += "### Feedback\n"
    description += "#### Strengths:\n" + "\n".join(f"- {strength}" for strength in feedback["strengths"]) + "\n"
    description += "#### Issues:\n" + "\n".join(f"- {issue}" for issue in feedback["issues"]) + "\n"
    
    return description



def fetch_metadata(metadata, keys):
    """Fetches specific keys from the metadata."""
    return {key: metadata.get(key, None) for key in keys}

def fetch_code_snippet(file_path, start_line, end_line):
    """Fetches a code snippet from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return ''.join(lines[start_line - 1:end_line])

class LLMReflector:
    def __init__(self, metadata, condensed_meta_log):
        self.metadata = metadata  # Rich metadata for the codebase
        self.condensed_meta_log = condensed_meta_log  # Pre-condensed metadata summary
        self.logs = []  # Logs of reflections, requests, and responses
        self.context = {}  # Current working context
        self.recursion_depth = 0

    def reflect(self, objective, context):
        """
        Reflects on the given context to determine if enough information exists to meet the objective.
        """
        reflection_prompt = f"""
Objective: {objective}

Current Context:
{context}

Condensed Meta Log:
{self.condensed_meta_log}

Do you have enough information to answer the objective? 
- If yes, provide the answer.
- If no, explain why and specify what additional information you need.
"""
        # Simulate LLM call
        response = call_llm(reflection_prompt)
        self.logs.append({"context": context, "response": response})
        return self.process_response(response)

    def process_response(self, response):
        """
        Processes the LLM's response to determine next steps.
        """
        if "I have enough information" in response:
            return {"status": "complete", "answer": response}
        else:
            additional_requests = self.parse_requests(response)
            return {"status": "incomplete", "requests": additional_requests}

    def parse_requests(self, response):
        """
        Extracts specific additional information requests from the LLM's response.
        """
        if "Please provide" in response:
            return [req.strip().strip('.') for req in response.split("Please provide")[1:]]
        return []

    def fetch_information(self, requests):
        """
        Fetches requested information from metadata or files.
        """
        for req in requests:
            if "CodeMapper" in req:
                code_mapper_data = self.metadata.get("files", {}).get("core/code_mapper.py", {}).get("classes", {}).get("CodeMapper", {})
                self.context["CodeMapper"] = code_mapper_data

            # Add logic for fetching other details (e.g., dependencies, functions, etc.)

    def run(self, objective):
        """
        Executes the iterative reflection loop.
        """
        # Step 1: Start with initial context from metadata and condensed meta log
        self.context = {
            "project_name": self.metadata.get("project", {}).get("name"),
            "project_description": self.metadata.get("project", {}).get("description"),
            "files": list(self.metadata.get("files", {}).keys())[:5],  # High-level file structure
        }

        # Add the condensed meta log to the initial context
        self.context["condensed_meta_log"] = self.condensed_meta_log

        while self.recursion_depth < 10:  # Prevent infinite loops
            result = self.reflect(objective, self.context)
            if result["status"] == "complete":
                return result["answer"]

            # Fetch additional information based on LLM requests
            self.fetch_information(result["requests"])
            self.recursion_depth += 1

        return "Failed to complete the task within recursion limits."

    # def call_llm(self, prompt):
    #     """
    #     Placeholder for LLM call (replace with actual API call).
    #     """
    #     return "Simulated LLM response: I need more information about the 'CodeMapper' class."

def iterative_codebase_description(metadata, condensed_meta_log):
    # Initialize reflector
    reflector = LLMReflector(metadata, condensed_meta_log)

    # Define the objective
    objective = (
        "Describe this codebase in detail, covering the major elements, "
        "its intent, and any feedback about its design and structure."
    )

    # Run the iterative loop
    result = reflector.run(objective)
    
    # Log and return the final result
    print("\nLogs of Reflections and Actions:")
    for log in reflector.logs:
        print(log)
    return result



if __name__ == "__main__":
    # Step 1: First time setup - Map and save the codebase metadata
    root_directory = "."
    ignore_files = {'.env', 'meta_log.json', 'discussion.txt', 'reviews.json', 'custom_mapper.py'}
    
    # Either generate new metadata...
    metadata = map_directory_structure_with_cross_references(root_directory, ignore_files=ignore_files)
    save_metadata_to_file(metadata)
    print(f"Full metadata saved to {OUTPUT_FILE}")
    
    # ...or load existing metadata
    # metadata = json.load(open("meta_log_20241125_154633.json"))
    condensed_metadata = condense_metadata(metadata)

    iterative_codebase_description = iterative_codebase_description(metadata, condensed_metadata)
    print(iterative_codebase_description)

    # Save to file with timestamp
    with open(f"codebase_description_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(iterative_codebase_description)
