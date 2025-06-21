"""
A script to find and report on relative imports in the JanusAI project,
suggesting absolute import paths from the 'janus_ai' package root.

This script can run in dry-run mode (default) or apply changes directly.

Usage:
    Dry run: python JanusAI/scripts/convert_relative_imports.py
    Apply changes: python JanusAI/scripts/convert_relative_imports.py --apply
"""

import os
import re
from pathlib import Path
import argparse # For --apply flag

# The name of the top-level package, as it would be imported.
PACKAGE_NAME = "janus_ai"
# The main source directory of the package.
SOURCE_DIR_NAME = "JanusAI"

def get_absolute_path_from_relative(current_file_path: Path, relative_import: str, project_root: Path) -> str:
    """
    Converts a relative import path to an absolute one based on the file's location.
    This version uses a more robust path resolution logic anchored to the project root.

    Args:
        current_file_path: The Path object of the file containing the import.
        relative_import: The relative import string (e.g., '.schemas' or '..core.grammar').
        project_root: The root directory of the project.

    Returns:
        The calculated absolute import string, or an error message.
    """
    # 1. Determine the number of levels to go up (count the leading dots)
    level = 0
    temp_path = relative_import
    while temp_path.startswith('.'):
        level += 1
        temp_path = temp_path[1:] # The part after the dots

    # 2. Start from the directory of the current file, resolved to an absolute path
    import_base_dir = current_file_path.parent.resolve()

    # 3. Move up the directory tree 'level' times
    for _ in range(level):
        import_base_dir = import_base_dir.parent

    # 4. Append the rest of the module path (the part after the dots)
    # temp_path is like 'schemas' or 'core.grammar'
    target_module_path_parts = []
    if temp_path: # Ensure temp_path is not empty (e.g. for 'from . import foo')
        target_module_path_parts = temp_path.split('.')

    # Construct the absolute file system path to the target module/package directory
    target_fs_path = import_base_dir.joinpath(*target_module_path_parts)

    # 5. Make the target_fs_path relative to the project_root
    try:
        # Resolve target_fs_path to ensure it's canonical before making relative
        # This path should represent the Python module path from the project root
        path_from_project_root = target_fs_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        # This error means target_fs_path is not under project_root, which is unexpected for valid relative imports.
        return f"# [ERROR] Resolved path '{target_fs_path.resolve()}' is outside project root '{project_root.resolve()}' for file '{current_file_path}' trying to import '{relative_import}'"

    # 6. Construct the final absolute import string using PACKAGE_NAME
    # path_from_project_root.parts will be like ('JanusAI', 'core', 'module') or ('tests', 'test_module')

    import_parts = list(path_from_project_root.parts)

    # If the first part of this path is the SOURCE_DIR_NAME, replace it with PACKAGE_NAME
    # e.g., ('JanusAI', 'core', 'module') -> ('janus_ai', 'core', 'module')
    if import_parts and import_parts[0] == SOURCE_DIR_NAME:
        final_module_path_parts = [PACKAGE_NAME] + import_parts[1:]
    # If the first part is something else (like 'tests'), prepend PACKAGE_NAME
    # e.g., ('tests', 'test_module') -> ('janus_ai', 'tests', 'test_module')
    # This assumes tests are also part of the overall 'janus_ai' import namespace.
    else:
        final_module_path_parts = [PACKAGE_NAME] + import_parts

    return ".".join(final_module_path_parts)


def analyze_file(file_path: Path, project_root: Path, apply_changes: bool = False):
    """
    Analyzes a single Python file for relative imports.
    If apply_changes is True, modifies the file in place.
    Otherwise, prints proposed changes (dry run).
    Returns a tuple: (number_of_lines_changed, True if file_was_changed else False)
    """
    lines_changed_count = 0
    file_was_changed = False

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        if not apply_changes:
            print(f"--- Could not read file: {file_path} ({e})")
        return 0, False

    new_lines = []
    relative_import_regex = re.compile(r"^(from\s+(\.+[a-zA-Z0-9_.]*)\s+import.*)")

    # header_printed_for_this_file can be local to the function call for dry run
    header_printed_for_this_file = False

    for line_number, line in enumerate(lines):
        match = relative_import_regex.match(line)
        if match:
            original_import_statement = match.group(1).strip()
            relative_path_str = match.group(2).strip()

            leading_whitespace = line[:match.start()]
            trailing_content = line[match.end():].rstrip('\n')

            parts = original_import_statement.split("import", 1) # Split only on the first "import"
            import_part_text = "import" + parts[1]

            absolute_path = get_absolute_path_from_relative(file_path, relative_path_str, project_root)

            if "[ERROR]" not in absolute_path:
                new_from_part = f"from {absolute_path}"
                new_import_statement = f"{new_from_part} {import_part_text}"
                new_line_content = f"{leading_whitespace}{new_import_statement}{trailing_content}\n"

                if new_line_content.strip() != line.strip():
                    if not apply_changes: # Dry run: print changes
                        if not header_printed_for_this_file:
                            print(f"\n--- Analyzing: {file_path} ---")
                            header_printed_for_this_file = True
                        print(f"  - {line.strip()}")
                        print(f"  + {new_line_content.strip()}")

                    new_lines.append(new_line_content)
                    lines_changed_count += 1
                    file_was_changed = True
                else:
                    new_lines.append(line)
            else: # Error in path conversion
                if not apply_changes: # Dry run: print error
                    if not header_printed_for_this_file:
                        print(f"\n--- Analyzing: {file_path} ---")
                        header_printed_for_this_file = True
                    print(f"  - {line.strip()}")
                    print(f"  + {absolute_path}")
                new_lines.append(line) # Keep original line on error
        else:
            new_lines.append(line)

    if file_was_changed:
        if apply_changes:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                # print(f"  Modified: {file_path} ({lines_changed_count} lines changed)")
            except Exception as e:
                print(f"--- Could not write to file: {file_path} ({e})")
                return 0, False # Failed to write
        elif header_printed_for_this_file: # Dry run separator only if header was printed
            print("-" * (len(str(file_path)) + 20))

    return lines_changed_count, file_was_changed


def main():
    parser = argparse.ArgumentParser(description="Convert relative imports to absolute imports for JanusAI project.")
    parser.add_argument("--apply", action="store_true", help="Apply changes directly to files. Default is dry run.")
    args = parser.parse_args()

    project_root = Path.cwd()

    if not (project_root / SOURCE_DIR_NAME).is_dir():
        print(f"Error: Source directory '{SOURCE_DIR_NAME}' not found in current directory '{project_root}'.")
        print("Please run this script from the root of the JanusAI project.")
        return

    APPLY_CHANGES_MODE = args.apply

    if APPLY_CHANGES_MODE:
        print(f"Starting CONVERSION of relative imports...")
    else:
        print(f"Starting analysis of relative imports (Dry Run)...")

    print(f"Scanning Python files in: {project_root}")
    print(f"Assuming package name: '{PACKAGE_NAME}' and source directory: '{SOURCE_DIR_NAME}'")

    # Removed static variable reset from main as it's now local to analyze_file for dry run

    total_files_changed_count = 0
    total_lines_transformed_count = 0

    for root_str, dirs, files in os.walk(project_root):
        root_path = Path(root_str)
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git', '.github', 'docs'] and not d.endswith('.egg-info')]

        for file_name in files:
            if file_name.endswith(".py"):
                file_path = root_path / file_name
                lines_transformed, file_updated = analyze_file(file_path, project_root, APPLY_CHANGES_MODE)
                if file_updated:
                    total_files_changed_count +=1
                    total_lines_transformed_count += lines_transformed

    if APPLY_CHANGES_MODE:
        print("\nConversion complete.")
        print(f"Total files modified: {total_files_changed_count}")
        print(f"Total lines modified: {total_lines_transformed_count}")
    else:
        print("\nDry run analysis complete.")

if __name__ == "__main__":
    main()
