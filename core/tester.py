import ast

def generate_tests(code):
    """Generate unit tests for the given code."""
    tree = ast.parse(code)
    test_code = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            test_code.append(f"def test_{node.name}():\n    # TODO: Implement test for {node.name}\n    pass\n")
    
    return "\n".join(test_code)
