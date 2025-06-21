import sys
import importlib
from typing import Any, Optional, Dict, Tuple
import logging

def safe_import(module_name: str, attribute: str = None) -> Tuple[bool, Any]:
    """Safely import a module or attribute with error handling."""
    try:
        module = importlib.import_module(module_name)
        if attribute:
            return True, getattr(module, attribute)
        return True, module
    except ImportError as e:
        logging.warning(f"Failed to import {module_name}: {e}")
        return False, None
    except AttributeError as e:
        logging.warning(f"Failed to get {attribute} from {module_name}: {e}")
        return False, None

def safe_env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    """Safely reset environment with proper error handling."""
    try:
        if seed is not None:
            if hasattr(env, 'reset'):
                return env.reset(seed=seed)
            else:
                # Fallback for older gym versions
                env.seed(seed)
                return env.reset(), {}
        else:
            if hasattr(env, 'reset'):
                result = env.reset()
                if isinstance(result, tuple):
                    return result
                return result, {}
            return None, {}
    except Exception as e:
        logging.error(f"Environment reset failed: {e}")
        return None, {}

def validate_inputs(**kwargs) -> bool:
    """Generic input validation utility."""
    for key, value in kwargs.items():
        if value is None:
            logging.error(f"Required input '{key}' is None")
            return False
    return True
