# File: JanusAI/scripts/setup_dev_env.py
#!/usr/bin/env python3
"""
setup_dev_env.py
===============

Quick setup script for Janus development environment.
Checks system requirements and sets up the development environment.
"""

import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        return False


def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required!")
        return False
    return True


def check_git():
    """Check if git is installed."""
    return run_command("git --version", check=False)


def create_venv():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists.")
        return True
    
    print("Creating virtual environment...")
    return run_command(f"{sys.executable} -m venv venv")


def activate_venv_message():
    """Print activation instructions based on OS."""
    if platform.system() == "Windows":
        print("\nTo activate the virtual environment, run:")
        print("  .\\venv\\Scripts\\activate")
    else:
        print("\nTo activate the virtual environment, run:")
        print("  source venv/bin/activate")


def upgrade_pip():
    """Upgrade pip to latest version."""
    print("\nUpgrading pip...")
    pip_cmd = "pip" if platform.system() == "Windows" else "pip3"
    return run_command(f"{pip_cmd} install --upgrade pip")


def install_dependencies():
    """Install project dependencies."""
    print("\nInstalling dependencies...")
    pip_cmd = "pip" if platform.system() == "Windows" else "pip3"
    
    # Check if pyproject.toml exists
    if Path("pyproject.toml").exists():
        print("Installing from pyproject.toml...")
        return run_command(f"{pip_cmd} install -e '.[dev]'")
    else:
        print("Installing from requirements files...")
        return run_command(f"{pip_cmd} install -r requirements-dev.txt")


def setup_precommit():
    """Set up pre-commit hooks."""
    print("\nSetting up pre-commit hooks...")
    if not run_command("pre-commit --version", check=False):
        print("pre-commit not found, skipping hook setup.")
        print("You can set it up later with: pre-commit install")
        return True
    
    return run_command("pre-commit install")


def check_optional_tools():
    """Check for optional but recommended tools."""
    print("\nChecking optional tools...")
    
    tools = {
        "make": "make --version",
        "docker": "docker --version",
        "nvidia-smi": "nvidia-smi --version",
    }
    
    for tool, cmd in tools.items():
        if run_command(cmd, check=False):
            print(f"✓ {tool} is available")
        else:
            print(f"✗ {tool} is not available (optional)")


def main():
    """Main setup function."""
    print("Janus Development Environment Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_git():
        print("ERROR: Git is required!")
        sys.exit(1)
    
    # Create virtual environment
    if not create_venv():
        print("ERROR: Failed to create virtual environment!")
        sys.exit(1)
    
    # Print activation instructions
    activate_venv_message()
    
    # Ask user if they want to continue with installation
    print("\nThis script will install development dependencies.")
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    # Note: The following commands should ideally be run inside the venv
    # For now, we'll just provide instructions
    print("\n" + "="*50)
    print("Please activate the virtual environment and run:")
    print("  python scripts/setup_dev_env.py --continue")
    print("\nOr manually run:")
    print("  pip install --upgrade pip")
    print("  pip install -e '.[dev]'  # or pip install -r requirements-dev.txt")
    print("  pre-commit install")
    print("="*50)
    
    # If --continue flag is passed, we assume we're in the venv
    if len(sys.argv) > 1 and sys.argv[1] == "--continue":
        if not upgrade_pip():
            print("WARNING: Failed to upgrade pip")
        
        if not install_dependencies():
            print("ERROR: Failed to install dependencies!")
            sys.exit(1)
        
        if not setup_precommit():
            print("WARNING: Failed to set up pre-commit hooks")
        
        check_optional_tools()
        
        print("\n" + "="*50)
        print("✓ Development environment setup complete!")
        print("\nNext steps:")
        print("  1. Run tests: make test-fast")
        print("  2. Format code: make format")
        print("  3. Run checks: make check")
        print("  4. See DEVELOPMENT.md for more information")
        print("="*50)


if __name__ == "__main__":
    main()