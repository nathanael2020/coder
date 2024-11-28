import pytest
import os
from custom_mapper import (
    chunk_file_syntax_aware,
    map_codebase,
    extract_metadata_from_file
)

# Fixtures for test files
@pytest.fixture
def sample_python_files(tmp_path):
    """Create a temporary directory with sample Python files"""
    # Create main file
    main_file = tmp_path / "main.py"
    main_file.write_text("""
def main():
    '''Main entry point'''
    print("Hello")
    helper_function()

def helper_function():
    '''Helper function'''
    return True
""")

    # Create a class file
    class_file = tmp_path / "my_class.py"
    class_file.write_text("""
class MyClass:
    '''A sample class'''
    def __init__(self):
        self.value = 42
        
    def get_value(self):
        '''Get the value'''
        return self.value
""")

    return tmp_path

# def test_chunk_file_syntax_aware(sample_python_files):
#     """Test that we can correctly identify functions and classes"""
#     main_file = sample_python_files / "main.py"
#     chunks = chunk_file_syntax_aware(str(main_file))
    
#     # Check we found both functions
#     assert len(chunks) == 2
#     function_names = {chunk["name"] for chunk in chunks}
#     assert function_names == {"main", "helper_function"}
    
#     # Check chunk structure
#     for chunk in chunks:
#         assert "id" in chunk
#         assert "type" in chunk
#         assert "content" in chunk
#         assert "start_line" in chunk
#         assert "end_line" in chunk

def test_extract_metadata(sample_python_files):
    """Test metadata extraction from files"""
    class_file = sample_python_files / "my_class.py"
    metadata = extract_metadata_from_file(str(class_file))
    
    # Check class metadata
    assert "MyClass" in metadata["classes"]
    class_meta = metadata["classes"]["MyClass"]
    
    # Check method metadata
    assert "get_value" in class_meta["methods"]
    assert class_meta["methods"]["get_value"]["description"] == "Get the value"

def test_map_codebase(sample_python_files):
    """Test mapping the entire codebase"""
    file_map = map_codebase(str(sample_python_files))
    
    # Check we found both files
    assert len(file_map) == 2
    
    # Check file paths
    file_paths = set(os.path.basename(path) for path in file_map.keys())
    assert file_paths == {"main.py", "my_class.py"}
    
    # Check content of mapped files
    for chunks in file_map.values():
        assert isinstance(chunks, list)
        assert all("id" in chunk for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)

def test_map_codebase_ignores_files(sample_python_files):
    """Test that mapping ignores specified files"""
    # Create a file that should be ignored
    ignore_file = sample_python_files / "meta_log.json"
    ignore_file.write_text("{}")
    
    file_map = map_codebase(str(sample_python_files))
    
    # Check that ignored file is not in map
    mapped_files = set(os.path.basename(path) for path in file_map.keys())
    assert "meta_log.json" not in mapped_files

# @pytest.mark.parametrize("filename,should_be_mapped", [
#     ("test.py", True),
#     ("test.js", True),
#     ("test.ts", True),
#     ("test.java", True),
#     ("test.txt", False),
#     ("test.md", False),
# ])

# def test_map_codebase_file_types(sample_python_files, filename, should_be_mapped):
#     """Test that mapping handles different file types correctly"""
#     test_file = sample_python_files / filename
#     test_file.write_text("// Test content")
    
#     file_map = map_codebase(str(sample_python_files))
#     is_mapped = any(filename in path for path in file_map.keys())
    
#     assert is_mapped == should_be_mapped