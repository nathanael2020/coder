from pathlib import Path
import git
from git import Repo
import tempfile
import shutil
from typing import Union, Dict, List, Optional
import os
from datetime import datetime
from core.code_mapper import CodeMapper
from utils.config import logger
import json
import docker
from typing import Tuple
import ast
import traceback
import uuid

from utils.config import IGNORE_PATHS
from utils.index import IndexManager

# Global instance
_codebase_manager_instance = None

def get_codebase_manager(workspace_dir: str = None) -> 'CodebaseManager':
    """Get or create the singleton instance of CodebaseManager."""
    global _codebase_manager_instance
    
    logger.info(f"get_codebase_manager called with workspace_dir: {workspace_dir}")
    logger.info(f"Current _codebase_manager_instance: {_codebase_manager_instance}")
    logger.info(f"Instance attributes: {vars(_codebase_manager_instance) if _codebase_manager_instance else None}")
    
    if _codebase_manager_instance is None:
        if workspace_dir is None:
            raise ValueError("workspace_dir must be provided when creating new CodebaseManager instance")
        _codebase_manager_instance = CodebaseManager(workspace_dir)
        logger.info(f"Created new CodebaseManager instance: {_codebase_manager_instance}")
    elif workspace_dir is not None:
        _codebase_manager_instance.set_workspace(workspace_dir)
        logger.info(f"Updated workspace for existing instance: {_codebase_manager_instance}")
    
    return _codebase_manager_instance

class CodebaseManager:
    """Manages codebase analysis, mapping, and modifications."""
    
    def __init__(self, workspace_dir: str = None):
        """Initialize with optional workspace_dir - can be set later."""
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.repo_dir = None
        self.repo_name = None
        self.code_mapper = None
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialized CodebaseManager with workspace: {self.workspace_dir}")
        self.index_manager = None
        self._active_dir = None

    def set_workspace(self, workspace_dir: str):
        """Set or update the workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        logger.info(f"Updated workspace directory to: {self.workspace_dir}")


    # def import_codebase(self, source: str) -> bool:

    #     logger.info(f"Importing codebase from: {source}")

    #     """Import a codebase from a git repository."""
    #     if not self.workspace_dir:
    #         self.workspace_dir = Path("sandbox")
            
    #     # Ensure workspace directory exists
    #     workspace_path = self.workspace_dir
    #     workspace_path.mkdir(parents=True, exist_ok=True)
            
    #     # Create a unique directory for this repository
    #     self.repo_name = source.split('/')[-1].replace('.git', '')
    #     self.repo_dir = workspace_path / self.repo_name
    #     logger.info(f"Setting repo_dir to: {self.repo_dir}")

    #     # Clone the repository
    #     if not self._clone_repository(source):
    #         return False

    #     # Save repository information
    #     self.save_repo_info(source)

    #     # Initialize code mapper after successful clone
    #     self.code_mapper = CodeMapper(self.repo_dir)
        
    #     # Initialize and run indexing
    #     # try:

    #     # Initialize index manager with self
    #     from utils.index import get_index_manager
    #     self.index_manager = get_index_manager(self)
    #     logger.info(f"Index manager initialized for repo: {self.repo_name}")
    #     logger.info(f"Index manager initialized: {self.index_manager}")
    
    #     logger.info(f"Repository imported successfully. Final state - repo_dir: {self.repo_dir}, repo_name: {self.repo_name}, index_manager: {self.index_manager}")
            
    #     logger.info(f"Repository imported successfully to: {self.repo_dir}")
    #     return True

    #     # except Exception as e:
    #     #     logger.error(f"Failed to import codebase: {e}")
    #     #     self.repo_dir = None  # Reset on failure
    #     #     return False

    def import_codebase(self, source: str) -> bool:
        """Import a codebase from a git repository."""
        logger.info(f"Starting import process for: {source}")
        logger.info(f"Initial state: {vars(self)}")
        
        if not self.workspace_dir:
            self.workspace_dir = Path("sandbox")
            
        # Ensure workspace directory exists
        workspace_path = self.workspace_dir
        workspace_path.mkdir(parents=True, exist_ok=True)
            
        # Create a unique directory for this repository
        self.repo_name = source.split('/')[-1].replace('.git', '')
        self.repo_dir = workspace_path / self.repo_name
        logger.info(f"Set repo_dir to: {self.repo_dir}")

        # Clone the repository
        if not self._clone_repository(source):
            return False

        # Initialize code mapper after successful clone
        self.code_mapper = CodeMapper(self.repo_dir)
        
        # Initialize and run indexing
        from utils.index import get_index_manager
        self.index_manager = get_index_manager(self)
        logger.info(f"Created index manager: {self.index_manager}")
        
        logger.info(f"Final state after import: {vars(self)}")
        return True

    def _clone_repository(self, git_url: str) -> bool:
        """Clone a git repository to the workspace."""
        try:
            logger.info(f"Cloning {git_url} to {self.workspace_dir}")
            
            # Extract repository name from URL
            repo_name = git_url.split('/')[-1].replace('.git', '')
            self.repo_dir = self.workspace_dir / repo_name
            
            # Check if repo already exists
            if self.repo_dir.exists():
                # Try to open existing repo
                try:
                    existing_repo = Repo(self.repo_dir)
                    # Check if it's the same remote URL
                    existing_remote = existing_repo.remote().url
                    if existing_remote == git_url:
                        logger.info(f"Repository {repo_name} already exists and matches remote URL. Pulling latest changes...")
                        # Pull latest changes
                        existing_repo.remote().pull()
                        # Save repo info
                        self._save_repo_info(git_url, repo_name, existing_repo)
                        return True
                    else:
                        logger.error(f"Directory exists with different remote: {existing_remote}")
                        raise ValueError(f"Directory {repo_name} exists but points to different remote: {existing_remote}")
                except git.InvalidGitRepositoryError:
                    logger.error(f"Directory {repo_name} exists but is not a git repository")
                    raise ValueError(f"Directory {repo_name} exists but is not a git repository")
            
            # Clone new repository if it doesn't exist
            repo = Repo.clone_from(git_url, self.repo_dir)
            
            # Save repository information
            self._save_repo_info(git_url, repo_name, repo)
            
            return True
        except Exception as e:
            logger.error(f"Git clone failed: {e}")
            return False
            
    def get_repo_dir(self) -> Path:
        """Get the path to the currently loaded repository."""
        return self.repo_dir
    
    def _save_repo_info(self, git_url: str, repo_name: str, repo: Repo):
        """Save repository information for future reference."""
        try:
            # Get git statistics
            commits = list(repo.iter_commits())
            contributors = set(commit.author.email for commit in commits)
            
            info = {
                "git_url": git_url,
                "repo_name": repo_name,
                "cloned_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "git_stats": {
                    "total_commits": len(commits),
                    "contributors": len(contributors),
                    "last_commit": commits[0].hexsha if commits else None,
                    "last_commit_date": commits[0].committed_datetime.isoformat() if commits else None,
                    "branches": [b.name for b in repo.branches],
                    "active_branch": repo.active_branch.name
                }
            }
            
            info_file = self.workspace_dir / "repo_info.json"
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving repo info: {e}")
            
    def _copy_local_directory(self, source_dir: Path) -> bool:
        """Copy local directory to workspace."""
        try:
            if not source_dir.exists():
                raise FileNotFoundError(f"Directory not found: {source_dir}")
                
            # Create a new directory for this import
            import_name = source_dir.name
            target_dir = self.workspace_dir / import_name
            
            shutil.copytree(
                source_dir,
                target_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git', '.env', 'venv', '.venv')
            )
            
            # Save import information
            self._save_import_info(str(source_dir), import_name)
            
            return True
        except Exception as e:
            logger.error(f"Directory copy failed: {e}")
            return False
            
    def _save_import_info(self, source_path: str, import_name: str):
        """Save information about imported local directory."""
        info = {
            "source_path": source_path,
            "import_name": import_name,
            "imported_at": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        info_file = self.workspace_dir / "import_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def modify_file(self, file_path: str, new_content: str) -> bool:

        """Modify a file in the workspace.
        
        Args:
            file_path (str): Path to the file relative to workspace
            new_content (str): New content for the file
            
        Returns:
            bool: True if modification was successful
        """
        try:
            full_path = self.workspace_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Create backup
            backup_path = full_path.with_suffix(f"{full_path.suffix}.bak")
            shutil.copy2(full_path, backup_path)
            
            # Write new content
            with open(full_path, 'w') as f:
                f.write(new_content)
                
            return True
        except Exception as e:
            logger.error(f"Failed to modify file {file_path}: {e}")
            return False

    def create_file(self, file_path: str, content: str) -> bool:
        """Create a new file in the workspace.
        
        Args:
            file_path (str): Path to the new file relative to workspace
            content (str): Content for the new file
            
        Returns:
            bool: True if file creation was successful
        """
        try:
            full_path = self.workspace_dir / file_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_path.exists():
                raise FileExistsError(f"File already exists: {file_path}")
                
            with open(full_path, 'w') as f:
                f.write(content)
                
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False

    def get_file_content(self, file_path: str) -> str:
        """Get content of a specific file from the workspace."""
        try:
            # Use repo_dir instead of workspace_dir
            full_path = self.repo_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(full_path) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
        
    def get_index_manager(self) -> Optional['IndexManager']:
        """Get or create the IndexManager instance."""
        logger.info(f"Getting IndexManager. Current state: {self.index_manager}")
        
        if not hasattr(self, 'index_manager') or self.index_manager is None:
            if not self.repo_dir or not self.workspace_dir:
                logger.error("Cannot create IndexManager: repository not imported")
                return None
                
            try:
                from utils.index import get_index_manager
                self.index_manager = get_index_manager(self)
                logger.info(f"Created new IndexManager instance for repo: {self.repo_name}")
            except Exception as e:
                logger.error(f"Error creating IndexManager: {e}")
                return None
                
        return self.index_manager    
    
    def get_workspace_path(self) -> Path:
        """Get the path to the current workspace."""
        return self.workspace_dir

    def cleanup(self, keep_workspace: bool = True):
        """Clean up workspace directory.
        
        Args:
            keep_workspace (bool): If True, keeps the workspace directory
        """
        try:
            if not keep_workspace:
                shutil.rmtree(self.workspace_dir)
            logger.info(f"Workspace {'kept' if keep_workspace else 'removed'}: {self.workspace_dir}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def analyze_codebase(self) -> Dict:
        """Analyze the codebase structure and dependencies."""

        self._active_dir = self.repo_dir
        
        try:
            self._validate_active_directory()
            logger.info("Starting codebase analysis...")
            
            # Get file tree separately
            file_tree = self.get_file_tree()
            
            # Generate metadata and stats
            analysis = {
                'files': {
                    'total': 0,
                    'by_type': {},
                    'by_size': {'small': 0, 'medium': 0, 'large': 0}
                },
                'code': {
                    'total_lines': 0,
                    'code_lines': 0,
                    'comment_lines': 0,
                    'blank_lines': 0,
                    'average_file_size': 0
                },
                'complexity': {
                    'total_functions': 0,
                    'total_classes': 0,
                    'average_function_length': 0,
                    'max_function_length': 0
                },
                'git': {},
                'metadata': {},
                'dependencies': {}
            }

            # Get the correct directory to analyze
            search_dir = self._active_dir
            logger.info(f"Analyzing directory: {search_dir}")

            total_chars = 0
            concatenated_code = []
            
            metadata = self.code_mapper.generate_metadata()
            analysis['metadata'] = metadata
            analysis['dependencies'] = metadata.get('dependencies', {})
            
            # Walk through the directory
            for root, _, files in os.walk(search_dir):
                for file in files:
                    if file.startswith('.') or file.startswith('__pycache__'):
                        continue
                    
                    file_path = Path(root) / file
                    ext = file_path.suffix
                    
                    # Update file type count
                    analysis['files']['by_type'][ext] = analysis['files']['by_type'].get(ext, 0) + 1
                    analysis['files']['total'] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.splitlines()
                            
                            # Analyze file content
                            file_stats = self._analyze_file_content(lines)
                            
                            # Update code stats
                            for key in ['total_lines', 'code_lines', 'comment_lines', 'blank_lines']:
                                analysis['code'][key] += file_stats[key]
                            
                            # Categorize file by size
                            if file_stats['total_lines'] < 100:
                                analysis['files']['by_size']['small'] += 1
                            elif file_stats['total_lines'] < 500:
                                analysis['files']['by_size']['medium'] += 1
                            else:
                                analysis['files']['by_size']['large'] += 1
                                
                            # Collect code for LLM processing
                            if ext in ['.py', '.js', '.html', '.css', '.md']:
                                total_chars += len(content)
                                rel_path = str(file_path.relative_to(search_dir))
                                concatenated_code.append(f"### File: {rel_path}\n{content}\n")
                                
                            # Analyze Python files for complexity
                            if ext == '.py':
                                self._analyze_python_file(file_path, analysis)
                                
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue

            # Calculate averages
            if analysis['files']['total'] > 0:
                analysis['code']['average_file_size'] = analysis['code']['total_lines'] / analysis['files']['total']

            # Get git information
            try:
                if self.repo_dir and self.repo_dir.exists():
                    repo = git.Repo(self.repo_dir)
                    commits = list(repo.iter_commits())
                    analysis['git'] = {
                        'total_commits': len(commits),
                        'contributors': len(set(commit.author.email for commit in commits)),
                        'last_modified': commits[0].committed_datetime.isoformat() if commits else None,
                        'active_branch': repo.active_branch.name,
                        'branches': len(list(repo.branches)),
                        'has_uncommitted_changes': repo.is_dirty()
                    }
            except Exception as e:
                logger.error(f"Error getting git info: {e}")

            # Add LLM processing information
            analysis['llm_processing'] = {
                'total_characters': total_chars,
                'can_process_with_gpt4': total_chars < 100000,
                'concatenated_code': "\n".join(concatenated_code) if total_chars < 100000 else "Codebase too large for concatenation"
            }

            # Add enhanced statistics
            analysis['enhanced_stats'] = {
                'file_types': {ext: count for ext, count in analysis['files']['by_type'].items()},
                'code_to_comment_ratio': analysis['code']['code_lines'] / analysis['code']['comment_lines'] 
                    if analysis['code']['comment_lines'] > 0 else 0,
                'estimated_read_time': f"{analysis['code']['total_lines'] // 400} minutes",
            }

            # Merge the file tree with the analysis
            analysis.update(file_tree)
            
            logger.info(f"Analysis complete. Stats: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def _analyze_python_file(self, file_path: Path, analysis: Dict):
        """Analyze a Python file for complexity metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            analysis['complexity']['total_functions'] += len(functions)
            analysis['complexity']['total_classes'] += len(classes)
            
            # Calculate function lengths
            for func in functions:
                func_lines = func.end_lineno - func.lineno
                analysis['complexity']['max_function_length'] = max(
                    analysis['complexity']['max_function_length'],
                    func_lines
                )
                
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")

    def _analyze_file_content(self, lines: List[str]) -> Dict[str, int]:
        """Analyze file content for detailed statistics."""
        stats = {
            'total_lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0
        }
        
        in_multiline_comment = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                stats['blank_lines'] += 1
            elif line.startswith('"""') or line.startswith("'''"):
                in_multiline_comment = not in_multiline_comment
                stats['comment_lines'] += 1
            elif in_multiline_comment:
                stats['comment_lines'] += 1
            elif line.startswith('#'):
                stats['comment_lines'] += 1
            else:
                stats['code_lines'] += 1
        
        return stats

    def execute_code(self, file_path: str = None) -> Tuple[str, str]:
        """Execute code in a Docker container and return output."""
        try:
            client = docker.from_env()
            
            # Prepare Docker volume mapping
            workspace_mount = {
                str(self.workspace_dir): {
                    'bind': '/workspace',
                    'mode': 'rw'
                }
            }
            
            # Run the code in a Python container
            container = client.containers.run(
                'python:3.9',
                command=f'python {file_path if file_path else "main.py"}',
                volumes=workspace_mount,
                working_dir='/workspace',
                detach=True,
                remove=True
            )
            
            # Get output with timeout
            try:
                output = container.logs(stdout=True, stderr=True, timeout=30)
                return output.decode(), ''
            except Exception as e:
                return '', f'Execution error: {str(e)}'
                
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return '', f'Docker error: {str(e)}'

    def modify_and_execute(self, file_path: str, new_content: str) -> Dict:
        """Modify a file and execute it, returning results and stats."""
        try:
            # Backup original file
            if not self.modify_file(file_path, new_content):
                raise Exception("Failed to modify file")
            
            # Execute the modified code
            output, error = self.execute_code(file_path)
            
            # Get updated stats
            stats = self._generate_stats()
            
            return {
                "status": "success",
                "output": output,
                "error": error,
                "stats": stats,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Modify and execute failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_directory_structure(self) -> Dict:
        """Get a nested dictionary representing the directory structure."""
        def add_path(structure: Dict, path: Path) -> None:
            parts = list(path.parts)
            current = structure
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        structure = {}
        for root, dirs, files in os.walk(self.workspace_dir):
            # Skip hidden directories and files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.') and f.endswith('.py')]
            
            for file in files:
                file_path = Path(os.path.relpath(os.path.join(root, file), self.workspace_dir))
                add_path(structure, file_path)
                
        return structure

    def get_file_content(self, file_path: str) -> str:
        """Get content of a specific file from the workspace."""
        try:
            full_path = self.workspace_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(full_path) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def get_file_tree(self) -> Dict:
        """Generate a hierarchical file tree structure."""
        try:
            tree = {}
            search_dir = self.repo_dir if self.repo_dir and self.repo_dir.exists() else self.workspace_dir
            
            def add_to_tree(path: str, node: dict) -> None:
                parts = path.split('/')
                current = node
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = {}

            # Walk the directory and build tree
            for root, dirs, files in os.walk(search_dir):
                # Skip hidden directories and their contents
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                files = [f for f in files if not f.startswith('.') and not f.endswith('.pyc')]
                
                for file in files:
                    file_path = Path(root) / file
                    rel_path = str(file_path.relative_to(search_dir))
                    add_to_tree(rel_path, tree)

            logger.info(f"Generated file tree with {len(tree)} root items")
            return {'files': tree}
            
        except Exception as e:
            logger.error(f"Error generating file tree: {e}")
            return {'files': {}}

    def set_active_directory(self, directory: Union[str, Path]) -> bool:
        """Set the active directory for indexing operations.
        
        Args:
            directory (Union[str, Path]): Directory path relative to workspace
            
        Returns:
            bool: True if directory was set successfully
        """
        try:
            full_path = self.workspace_dir / Path(directory)
            if not full_path.exists() or not full_path.is_dir():
                raise ValueError(f"Invalid directory path: {directory}")
            
            self._active_dir = full_path
            logger.info(f"Active directory set to: {self._active_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to set active directory: {e}")
            return False
            
    def get_active_directory(self) -> Optional[Path]:
        """Get the current active directory for indexing."""
        return self._active_dir
        
    def _validate_active_directory(self):
        """Validate that an active directory has been set.
        
        Raises:
            ValueError: If no active directory has been set
        """
        if self._active_dir is None:
            raise ValueError("No active directory set. Call set_active_directory() first.")
        
    def save_repo_info(self, source: str):
        """Save repository information."""
        try:
            info = {
                "source": source,
                "session_id": self.session_id,
                "import_time": datetime.now().isoformat(),
                "repo_dir": str(self.repo_dir) if self.repo_dir else None
            }
            # Save info to a file or database as needed
            logger.info(f"Saved repo info: {info}")
        except Exception as e:
            logger.error(f"Error saving repo info: {e}")
            

    def should_ignore_path(self, path: str) -> bool:
        """Check if the given path should be ignored based on the IGNORE_PATHS list."""
        for ignore_path in IGNORE_PATHS:
            if path.startswith(ignore_path):
                return True
        return False
    