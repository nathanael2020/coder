"""
code_mapper.py

This module contains the CodeMapper class, which is responsible for mapping the codebase into functions, classes, and dependencies,
as well as facilitating NLP and LLM-driven code interactions.

"""

import os
import json
from hashlib import sha256
from datetime import datetime
import ast
from utils.config import logger, client
from pathlib import Path
from typing import Dict
class CodeMapper:
    """A class for analyzing and mapping code structure and dependencies.
    
    This class provides functionality to analyze Python codebases, generate metadata,
    and map dependencies between files and functions.
    
    Attributes:
        workspace_dir (str): Root directory of the codebase to analyze
        ignore_dirs (set): Directories to ignore during analysis
        ignore_files (set): Files to ignore during analysis
    """
    
    def __init__(self, repo_dir: str):
        """Initialize CodeMapper with workspace directory.
        
        Args:
            repo_dir (str): Path to the root directory of the codebase
        """    
        self.repo_dir = Path(repo_dir)
        self.workspace_dir = self.repo_dir.parent
        self.file_map = {}
        self.dependency_graph = {}
        self.map_codebase()
        # self.metadata = self.generate_metadata()

    def map_codebase(self):
        """Map the codebase starting from base_path."""
        for root, _, files in os.walk(self.repo_dir):
            # Skip hidden directories and virtual environments
            if any(part.startswith('.') or part == 'venv' or part == '__pycache__' 
                  for part in Path(root).parts):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_dir)
                    # Rest of the mapping logic...

    def generate_metadata(self) -> Dict:
        """Generate metadata for all files in the codebase."""
        metadata = {}
        dependencies = {}
        
        try:
            # Walk through the directory
            for root, _, files in os.walk(self.repo_dir):
                # Skip hidden directories and their contents
                if any(part.startswith('.') for part in Path(root).parts):
                    continue
                    
                for file in files:
                    # Skip hidden files, compiled Python files, and cache
                    if file.startswith('.') or file.endswith('.pyc') or '__pycache__' in root:
                        continue
                    
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(self.repo_dir)
                    
                    # Include all files in metadata
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            metadata[str(rel_path)] = {
                                'size': len(content),
                                'lines': len(content.splitlines()),
                                'type': file_path.suffix.lower() or 'no_extension',
                                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                            }
                            
                            # Only analyze Python files for dependencies
                            if file_path.suffix.lower() == '.py':
                                deps = self._extract_python_dependencies(content)
                                if deps:
                                    dependencies[str(rel_path)] = deps
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
            
            logger.info(f"Generated metadata for {len(metadata)} files")
            logger.info(f"File types found: {set(item['type'] for item in metadata.values())}")
            
            return {
                'metadata': metadata,
                'dependencies': dependencies
            }
            
        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            return {'metadata': {}, 'dependencies': {}}


    def _extract_python_dependencies(self, content: str) -> dict:
        """Extract imports and function dependencies from Python code.
        
        Args:
            content (str): Python file content
            
        Returns:
            dict: Dictionary containing imports and function dependencies
        """
        try:
            tree = ast.parse(content)
            dependencies = {
                'imports': [],
                'from_imports': {},
                'functions': set(),
                'classes': set()
            }
            
            for node in ast.walk(tree):
                # Handle regular imports
                if isinstance(node, ast.Import):
                    for name in node.names:
                        dependencies['imports'].append(name.name)
                        
                # Handle from imports
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.module not in dependencies['from_imports']:
                            dependencies['from_imports'][node.module] = []
                        dependencies['from_imports'][node.module].extend(
                            name.name for name in node.names
                        )
                        
                # Track function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dependencies['functions'].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        dependencies['functions'].add(node.func.attr)
                        
                # Track class definitions
                elif isinstance(node, ast.ClassDef):
                    dependencies['classes'].add(node.name)
            
            # Convert sets to lists for JSON serialization
            dependencies['functions'] = list(dependencies['functions'])
            dependencies['classes'] = list(dependencies['classes'])
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {e}")
            return {
                'imports': [],
                'from_imports': {},
                'functions': [],
                'classes': []
            }
    def _get_decorator_str(self, node: ast.expr) -> str:
        """Convert decorator AST node to string representation."""
        try:
            return ast.unparse(node)
        except:
            return str(node)

    def _get_base_str(self, node: ast.expr) -> str:
        """Convert base class AST node to string representation."""
        try:
            return ast.unparse(node)
        except:
            return str(node)

    def extract_metadata_from_file(self, file_path: str) -> dict:
        """Extract metadata from a single Python file.
        
        Args:
            file_path (str): Path to the Python file
            
        Returns:
            dict: File metadata including classes, functions, and dependencies
        """
        metadata = {
            "description": "",
            "entry_points": [],
            "themes": [],
            "imports": [],
            "classes": {},
            "functions": {},
            "has_main": False
        }
        
        try:
            with open(file_path, "r") as file:
                content = file.read()
                tree = ast.parse(content)

                # Extract module-level docstring
                metadata["description"] = ast.get_docstring(tree) or ""

                # Process the AST
                for node in tree.body:
                    # Check for main block
                    if (isinstance(node, ast.If) and 
                        isinstance(node.test, ast.Compare) and 
                        isinstance(node.test.left, ast.Name) and 
                        node.test.left.id == "__name__" and 
                        isinstance(node.test.comparators[0], ast.Constant) and 
                        node.test.comparators[0].value == "__main__"):
                        metadata["has_main"] = True
                        for n in ast.walk(node):
                            if isinstance(n, ast.Call):
                                if isinstance(n.func, ast.Name):
                                    metadata["entry_points"].append(n.func.id)
                                elif isinstance(n.func, ast.Attribute):
                                    metadata["entry_points"].append(n.func.attr)

                    # Extract imports
                    if isinstance(node, ast.Import):
                        metadata["imports"].extend(name.name for name in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        metadata["imports"].append(node.module)

                    # Process classes and functions
                    if isinstance(node, ast.ClassDef):
                        metadata["classes"][node.name] = self._process_class(node)
                    elif isinstance(node, ast.FunctionDef):
                        metadata["functions"][node.name] = self._process_function(node)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            metadata["error"] = str(e)

        return metadata

    def _process_class(self, node: ast.ClassDef) -> dict:
        """Process a class definition node.
        
        Args:
            node (ast.ClassDef): Class definition AST node
            
        Returns:
            dict: Class metadata including methods and dependencies
        """
        return {
            "line_range": [node.lineno, getattr(node, 'end_lineno', node.lineno)],
            "description": ast.get_docstring(node) or "",
            "methods": {
                item.name: self._process_function(item)
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
        }

    def _process_function(self, node: ast.FunctionDef) -> dict:
        """Process a function definition node.
        
        Args:
            node (ast.FunctionDef): Function definition AST node
            
        Returns:
            dict: Function metadata including parameters and dependencies
        """
        return {
            "line_range": [node.lineno, getattr(node, 'end_lineno', node.lineno)],
            "description": ast.get_docstring(node) or "",
            "parameters": [arg.arg for arg in node.args.args],
            "dependencies": self._extract_dependencies(node)
        }

    def _extract_dependencies(self, node: ast.AST) -> list:
        """Extract function call dependencies from an AST node.
        
        Args:
            node (ast.AST): AST node to analyze
            
        Returns:
            list: List of function names called within the node
        """
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.add(child.func.attr)
        return list(dependencies)

    def extract_imports(self, file_path: str) -> list:
        """Extract imports from a Python file.
        
        Args:
            file_path (str): Path to the Python file
            
        Returns:
            list: List of imported module names
        """
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(name.name for name in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module)
            return imports
        except Exception as e:
            logger.error(f"Error extracting imports from {file_path}: {e}")
            return []

# Usage example
if __name__ == "__main__":
    root_directory = "./"
    mapper = CodeMapper(root_directory)
    mapper.map_codebase()
    print(json.dumps(mapper.get_code_map(), indent=4))
    print(json.dumps(mapper.get_chunks(), indent=4))
