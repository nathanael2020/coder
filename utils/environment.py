import subprocess

def create_git_branch(branch_name):
    """Create a new Git branch."""
    try:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
        return {"success": True, "message": f"Branch '{branch_name}' created successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr}
