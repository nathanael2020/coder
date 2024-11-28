"""
environment.py

This module contains functions for interacting with the environment, such as creating a new Git branch.

"""

# Filestructure:
# coder_v2/
#   main.py
#   static/
#   logs/
#   utils/
#   core/

# File: utils/environment.py

import subprocess

def create_git_branch(branch_name):
    """Create a new Git branch."""
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        return {"success": True, "message": f"Branch '{branch_name}' created successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr}
