from core.code_mapper import CodeMapper
import json

class CodeConcierge:
    def __init__(self, code_mapper):
        self.code_mapper = code_mapper
        self.load_code_map()

    def load_code_map(self):
        """Load the existing code map or generate a new one if none exists."""
        try:
            with open('code_map.json', 'r') as infile:
                self.code_map = json.load(infile)
        except FileNotFoundError:
            self.code_mapper.map_codebase()
            self.code_map = self.code_mapper.get_code_map()

    def describe_function(self, function_name):
        """Describe the details of a specific function."""
        if function_name in self.code_map["functions"]:
            function_info = self.code_map["functions"][function_name]
            return f"Function '{function_name}' is defined in file '{function_info['file']}', starting at line {function_info['start_line']}."
        else:
            return f"Function '{function_name}' not found in the codebase."

    def describe_class(self, class_name):
        """Describe the details of a specific class."""
        if class_name in self.code_map["classes"]:
            class_info = self.code_map["classes"][class_name]
            return f"Class '{class_name}' is defined in file '{class_info['file']}', starting at line {class_info['start_line']}."
        else:
            return f"Class '{class_name}' not found in the codebase."

    def list_dependencies(self):
        """List all dependencies used in the codebase."""
        dependencies = ", ".join(self.code_map["dependencies"].keys())
        return f"The codebase has the following dependencies: {dependencies}"

# Usage example
mapper = CodeMapper(root_directory=".")
concierge = CodeConcierge(mapper)
print(concierge.describe_function("my_function"))
