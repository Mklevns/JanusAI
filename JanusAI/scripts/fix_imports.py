#!/usr/bin/env python
"""
fix_imports.py - A utility to correct import statements across the JanusAI project.

This script scans all Python files within the project and replaces any old
'JanusAI' import prefixes with the correct 'janus_ai' prefix. This is necessary
to align with standard Python packaging practices where the package name in
pyproject.toml (`janus-ai`) is normalized to `janus_ai` for imports.

Usage:
1. Run this script from the project root directory:
   python JanusAI/scripts/fix_imports.py
"""

import os

def fix_imports_in_project():
    """
    Walks through the project, finds all .py files, and corrects
    'from janus_ai' or 'import janus_ai' to 'from janus_ai' or 'import janus_ai'.
    """
    # Assuming the script is in JanusAI/scripts/fix_imports.py
    # project_root will be the directory containing the JanusAI directory.
    # This correctly identifies the project root based on the script's location.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # The source directory is JanusAI, located directly in the project_root
    janusai_source_dir = os.path.join(project_root, 'JanusAI')
    
    if not os.path.isdir(janusai_source_dir):
        print(f"❌ Error: 'JanusAI' source directory not found at expected location: {janusai_source_dir}")
        print("   Please ensure the script is located at 'JanusAI/scripts/fix_imports.py' within the project structure.")
        return
        
    print(f"🔍 Project root identified as: {project_root}")
    print(f"📂 Source directory for fixing imports: {janusai_source_dir}")

    # Define the transformations
    # Order matters if prefixes are substrings of each other, but not here.
    replacements_map = {
        "from janus_ai.": "from janus_ai.",
        "import janus_ai.": "import janus_ai.",
        "from janus_ai.": "from janus_ai.",
        "import janus_ai.": "import janus_ai.",
        # The problem description also implies replacing "janus_ai" (incorrect) and "JanusAI" (incorrect)
        # as whole words if they are package names.
        # The above handles module path style.
        # Let's consider if `import janus` or `import JanusAI` (without a trailing dot) is possible.
        # If `janus` or `JanusAI` are top-level packages, they'd be `import janus` or `from janus import ...`
        # The instructions were:
        # "replace all occurrences of from janus_ai. with from janus_ai."
        # "replace all occurrences of import janus_ai. with import janus_ai."
        # "replace all occurrences of from janus_ai. with from janus_ai."
        # "replace all occurrences of import janus_ai. with import janus_ai."
        # These are specific and include the dot, implying submodule access.
        # If there are instances of `import janus` or `import JanusAI` (as a whole package),
        # those would need `import janus_ai`.
        # The script should also handle `from janus import foo` -> `from janus_ai import foo`
        # and `import janus` -> `import janus_ai`.

        # Adding more general replacements for package names if they appear alone:
        "from janus import": "from janus_ai import",
        "import janus\n": "import janus_ai\n", # Ensure specificity for 'import janus'
        "import janus ": "import janus_ai ",   # Ensure specificity for 'import janus as ...'
        "from JanusAI import": "from janus_ai import",
        "import JanusAI\n": "import janus_ai\n",
        "import JanusAI ": "import janus_ai ",
    }
    # Add a check for `import janus` at the end of a line
    # For `import janus` as a whole line: content.replace('\nimport janus\n', '\nimport janus_ai\n')
    # This is getting complex. The original instructions were very specific with trailing dots.
    # I will stick to the original specification for direct replacements first.
    # The script can be enhanced later if other forms are found.

    # Re-evaluating the initial request:
    # "replace all occurrences of from janus_ai. with from janus_ai."
    # "replace all occurrences of import janus_ai. with import janus_ai."
    # "replace all occurrences of from janus_ai. with from janus_ai."
    # "replace all occurrences of import janus_ai. with import janus_ai."
    # These are clear. The script should implement these four.

    # The script's original (flawed) structure was:
    # old_prefix_from = "from janus_ai" -> should be "from janus_ai." or "from janus_ai."
    # new_prefix_from = "from janus_ai" -> should be "from janus_ai."
    # old_prefix_import = "import janus_ai" -> should be "import janus_ai." or "import janus_ai."
    # new_prefix_import = "import janus_ai" -> should be "import janus_ai."

    # Let's define the exact search and replace strings as per instructions for item 1.1
    # These are literal string replacements.
    replacements_definitions = [
        ("from janus_ai.", "from janus_ai."),
        ("import janus_ai.", "import janus_ai."),
        ("from janus_ai.", "from janus_ai."),
        ("import janus_ai.", "import janus_ai."),
    ]

    files_modified = 0
    total_replacements_count = 0
    
    print("="*70)
    print("🚀 Starting JanusAI Import Fixer...")
    print(f"Scanning for Python files in: {janusai_source_dir}")
    print("="*70)

    for root, _, files in os.walk(janusai_source_dir):
        # Exclude .venv and __pycache__ directories
        if '.venv' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replacements_in_file_count = 0
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    for old_import, new_import in replacements_definitions:
                        if old_import in content:
                            count = content.count(old_import)
                            content = content.replace(old_import, new_import)
                            replacements_in_file_count += count
                        
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        relative_path = os.path.relpath(file_path, project_root)
                        print(f"✅ Fixed {replacements_in_file_count} import string(s) in: {relative_path}")
                        files_modified += 1
                        total_replacements_count += replacements_in_file_count
                        
                except Exception as e:
                    print(f"⚠️  Could not process file {file_path}: {e}")

    print("\n" + "="*70)
    print("🎉 Scan complete!")
    if total_replacements_count > 0:
        print(f"   Total files modified: {files_modified}")
        print(f"   Total import strings corrected: {total_replacements_count}")
    else:
        print("   No matching import strings for replacement were found.")
    
    print("="*70)


if __name__ == "__main__":
    fix_imports_in_project()
