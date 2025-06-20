# JanusAI/scripts/validate_configs.py
#!/usr/bin/env python3
"""
validate_configs.py
==================

Validate YAML configuration files for the Janus project.
This script is used by pre-commit hooks to ensure config files are valid.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import ValidationError

# Try to import Janus config models
try:
    from janus.config.models import JanusConfig, ExperimentConfig
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: Janus config models not available. Basic YAML validation only.")


def validate_yaml_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate YAML file syntax."""
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, "OK"
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_config_schema(file_path: Path, config_data: dict) -> Tuple[bool, str]:
    """Validate config against Pydantic schema if available."""
    if not MODELS_AVAILABLE:
        return True, "Schema validation skipped (models not available)"
    
    try:
        # Determine config type based on file location or content
        if 'experiment' in str(file_path) or 'experiment_type' in config_data:
            ExperimentConfig(**config_data)
        else:
            JanusConfig(**config_data)
        return True, "Schema validation passed"
    except ValidationError as e:
        return False, f"Schema validation error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def check_required_fields(config_data: dict) -> Tuple[bool, str]:
    """Check for required fields in config."""
    # Basic required fields for most configs
    required_fields = []
    
    if 'experiment_type' in config_data:
        # Experiment config
        required_fields = ['experiment_type', 'algorithm']
    elif 'training' in config_data:
        # Training config
        required_fields = ['training']
    
    missing_fields = [field for field in required_fields if field not in config_data]
    
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    return True, "Required fields present"


def validate_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single configuration file."""
    errors = []
    
    # Check file exists
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    
    # Validate YAML syntax
    is_valid, message = validate_yaml_syntax(file_path)
    if not is_valid:
        errors.append(message)
        return False, errors
    
    # Load config data
    try:
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to load config: {e}")
        return False, errors
    
    # Check required fields
    is_valid, message = check_required_fields(config_data)
    if not is_valid:
        errors.append(message)
    
    # Validate schema
    is_valid, message = validate_config_schema(file_path, config_data)
    if not is_valid:
        errors.append(message)
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate Janus configuration files")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Configuration files to validate"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    all_valid = True
    
    for file_path in args.files:
        is_valid, errors = validate_file(file_path)
        
        if args.verbose or not is_valid:
            print(f"\n{'='*60}")
            print(f"Validating: {file_path}")
            print(f"{'='*60}")
        
        if is_valid:
            if args.verbose:
                print("✓ Validation passed")
        else:
            all_valid = False
            print("✗ Validation failed:")
            for error in errors:
                print(f"  - {error}")
    
    if not all_valid:
        sys.exit(1)
    elif args.verbose:
        print(f"\n✓ All {len(args.files)} files validated successfully")


if __name__ == "__main__":
    main()