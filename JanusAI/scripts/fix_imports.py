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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # --- ENHANCED DIRECTORY CHECKING ---
    # The source directory is JanusAI, located directly in the project_root
    janusai_dir = os.path.join(project_root, 'JanusAI')
    
    source_dir = None
    
    if os.path.isdir(janusai_dir):
        source_dir = janusai_dir
        print("🔍 Found 'JanusAI' directory. Will fix imports inside.")
    else:
        print(f"❌ Error: 'JanusAI' source directory not found in the project root.")
        print("   Please ensure you are running this script from the correct location relative to the 'JanusAI' directory.")
        return
        
    prefixes_to_replace = {
        "from janus_ai.": "from janus_ai.",
        "import janus_ai.": "import janus_ai.",
        "from janus_ai.": "from janus_ai.",
        "import janus_ai.": "import janus_ai."
    }
    
    files_modified = 0
    total_replacements = 0
    
    print("="*70)
    print("🚀 Starting JanusAI Import Fixer...")
    print(f"Scanning for Python files in: {source_dir}")
    print("="*70)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                replacements_in_file = 0
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    new_content = content
                    
                    for old_prefix, new_prefix in prefixes_to_replace.items():
                        if old_prefix in new_content:
                            count = new_content.count(old_prefix)
                            new_content = new_content.replace(old_prefix, new_prefix)
                            replacements_in_file += count
                        
                    if replacements_in_file > 0:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        relative_path = os.path.relpath(file_path, project_root)
                        print(f"✅ Fixed {replacements_in_file} import(s) in: {relative_path}")
                        files_modified += 1
                        total_replacements += replacements_in_file
                        
                except Exception as e:
                    print(f"⚠️  Could not process file {file_path}: {e}")

    print("\n" + "="*70)
    print("🎉 Scan complete!")
    if total_replacements > 0:
        print(f"   Total files modified: {files_modified}")
        print(f"   Total imports corrected: {total_replacements}")
    else:
        print("   No incorrect 'JanusAI' imports were found. Your project might already be using 'janus_ai' or has no 'JanusAI' imports.")
    
    print("="*70)


if __name__ == "__main__":
    fix_imports_in_project()
