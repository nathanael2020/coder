import ast
import os
import json

class CodeMapper:
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.code_map = {
            "functions": {},
            "classes": {},
            "dependencies": {}
        }

    def map_codebase(self):
        """Map the entire codebase into functions, classes, and dependencies."""
        for subdir, _, files in os.walk(self.root_directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(subdir, file)
                    with open(file_path, 'r') as f:
                        code = f.read()
                        tree = ast.parse(code)
                        self._map_file(tree, file_path)

        # Save the generated map to a file for persistence
        with open('code_map.json', 'w') as outfile:
            json.dump(self.code_map, outfile, indent=4)

    def _map_file(self, tree, file_path):
        """Map individual Python files for functions, classes, and imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.code_map["functions"][node.name] = {
                    "file": file_path,
                    "start_line": node.lineno
                }
            elif isinstance(node, ast.ClassDef):
                self.code_map["classes"][node.name] = {
                    "file": file_path,
                    "start_line": node.lineno
                }
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.code_map["dependencies"].setdefault(alias.name, []).append(file_path)

    def get_code_map(self):
        """Return the current code map."""
        return self.code_map
