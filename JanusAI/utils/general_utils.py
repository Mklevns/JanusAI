"""
General Utility Functions
=========================

This module contains various utility functions that can be used across
different parts of the Janus framework.
"""

from typing import Any

def safe_import(module_name: str, package_name: Optional[str] = None) -> Any:
    """
    Safely imports a module, returning None if the module is not found.
    Useful for optional dependencies.

    Args:
        module_name: The name of the module to import (e.g., 'wandb').
        package_name: The name of the package (e.g., 'wandb'). If not provided, defaults to module_name.

    Returns:
        The imported module object if successful, None otherwise.
    """
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Warning: Optional package '{package_name}' not found. Some functionality may be limited.")
        return None

