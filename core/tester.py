"""
tester.py

This module contains functionality for automatically generating unit tests for Python code.

Functions:
    generate_tests: Generates unit test skeletons for all functions in the provided code.

Dependencies:
    - ast: For parsing and analyzing Python code structure
"""

import ast
def generate_tests(code: str) -> str:
    """
    Generate unit test skeletons for all functions in the provided code.

    This function parses Python code using the ast module and creates test function
    templates for each function definition found in the code. The generated tests
    follow the naming convention 'test_<function_name>'.

    Args:
        code (str): The source code to generate tests for. Should be valid Python code.

    Returns:
        str: A string containing the generated test code, with one test function per
             function found in the input code. Each test function contains a TODO
             comment and a pass statement.

    Example:
        >>> code = '''
        ... def add(a, b):
        ...     return a + b
        ... '''
        >>> print(generate_tests(code))
        def test_add():
            # TODO: Implement test for add
            pass

    Note:
        - The generated tests are skeleton implementations that need to be filled in
        - Test names are automatically prefixed with 'test_' to be recognized by pytest
        - Each generated test includes a TODO comment indicating what needs to be tested
    """
    tree = ast.parse(code)
    test_code = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            test_code.append(f"def test_{node.name}():\n    # TODO: Implement test for {node.name}\n    pass\n")
    
    return "\n".join(test_code)
