"""
concierge.py

This module contains the Concierge class, which is responsible for managing the codebase and providing information about the codebase to the user.

"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: core/concierge.py

from core.code_mapper import CodeMapper
import json

class CodeConcierge:
    """
    A class that manages and provides information about the codebase.

    The CodeConcierge acts as a central point for accessing information about the code structure,
    including functions, classes, and dependencies. It maintains a code map that can be loaded
    from disk or generated on demand.

    Attributes:
        code_mapper (CodeMapper): An instance of CodeMapper used to analyze the codebase
        code_map (dict): A dictionary containing the mapped codebase information including:
            - functions: Dict mapping function names to their metadata
            - classes: Dict mapping class names to their metadata
            - dependencies: Dict of project dependencies

    Example:
        >>> mapper = CodeMapper(root_directory=".")
        >>> concierge = CodeConcierge(mapper)
        >>> print(concierge.describe_function("my_function"))
    """

    def __init__(self, code_mapper):
        """
        Initialize the CodeConcierge with a CodeMapper instance.

        Args:
            code_mapper (CodeMapper): An instance of CodeMapper used to analyze the codebase
        """
        self.code_mapper = code_mapper
        self.load_code_map()

    def load_code_map(self):
        """
        Load the existing code map from disk or generate a new one if none exists.

        The code map is loaded from 'code_map.json' if it exists. Otherwise, it triggers
        the CodeMapper to analyze the codebase and generate a new map.

        Raises:
            FileNotFoundError: Handled internally - triggers new map generation
            JSONDecodeError: Could occur if code_map.json is corrupted
        """
        try:
            with open('code_map.json', 'r') as infile:
                self.code_map = json.load(infile)
        except FileNotFoundError:
            self.code_mapper.map_codebase()
            self.code_map = self.code_mapper.get_code_map()

    def describe_function(self, function_name):
        """
        Get a description of a specific function's location and details.

        Args:
            function_name (str): Name of the function to describe

        Returns:
            str: A formatted string describing the function's location and details,
                or an error message if the function is not found

        Example:
            >>> concierge.describe_function("parse_error")
            "Function 'parse_error' is defined in file 'core/error_handler.py', starting at line 42."
        """
        if function_name in self.code_map["functions"]:
            function_info = self.code_map["functions"][function_name]
            return f"Function '{function_name}' is defined in file '{function_info['file']}', starting at line {function_info['start_line']}."
        else:
            return f"Function '{function_name}' not found in the codebase."

    def describe_class(self, class_name):
        """
        Get a description of a specific class's location and details.

        Args:
            class_name (str): Name of the class to describe

        Returns:
            str: A formatted string describing the class's location and details,
                or an error message if the class is not found

        Example:
            >>> concierge.describe_class("CodeConcierge")
            "Class 'CodeConcierge' is defined in file 'core/concierge.py', starting at line 15."
        """
        if class_name in self.code_map["classes"]:
            class_info = self.code_map["classes"][class_name]
            return f"Class '{class_name}' is defined in file '{class_info['file']}', starting at line {class_info['start_line']}."
        else:
            return f"Class '{class_name}' not found in the codebase."

    def list_dependencies(self):
        """
        Get a list of all dependencies used in the codebase.

        Returns:
            str: A formatted string listing all dependencies found in the codebase

        Example:
            >>> concierge.list_dependencies()
            "The codebase has the following dependencies: json, os, subprocess"
        """
        dependencies = ", ".join(self.code_map["dependencies"].keys())
        return f"The codebase has the following dependencies: {dependencies}"

# Usage example
mapper = CodeMapper(root_directory=".")
concierge = CodeConcierge(mapper)
print(concierge.describe_function("my_function"))
