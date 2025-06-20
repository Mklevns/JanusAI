#!/usr/bin/env python
"""
fix_imports.py - A utility to correct import statements across the JanusAI project.

This script scans all Python files within the project and replaces any old
'JanusAI' import prefixes with the correct 'janus' prefix. This is necessary
to resolve ModuleNotFoundError issues after renaming the main package directory.

Usage:
1. Run this script from the project root directory:
   python scripts/fix_imports.py
2. If the script ran on the 'JanusAI' directory, rename it to 'janus':
   mv JanusAI janus
"""

import os

def fix_imports_in_project():
    """
    Walks through the project, finds all .py files, and corrects
    'from JanusAI' or 'import JanusAI' to 'from janus' or 'import janus'.
    This version is more robust and checks for both 'janus' and 'JanusAI' directories.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- ENHANCED DIRECTORY CHECKING ---
    janus_dir = os.path.join(project_root, 'janus')
    janusai_dir = os.path.join(project_root, 'JanusAI')
    
    source_dir = None
    needs_rename = False
    
    if os.path.isdir(janus_dir):
        source_dir = janus_dir
        print("🔍 Found 'janus' directory. Scanning for old imports...")
    elif os.path.isdir(janusai_dir):
        source_dir = janusai_dir
        needs_rename = True
        print("🔍 Found 'JanusAI' directory. Will fix imports inside.")
    else:
        print(f"❌ Error: Neither 'janus' nor 'JanusAI' source directories found in the project root.")
        print("   Please ensure you are running this script from the correct location.")
        return
        
    old_prefix_from = "from JanusAI"
    new_prefix_from = "from janus"
    old_prefix_import = "import JanusAI"
    new_prefix_import = "import janus"
    
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
                    
                    # Fix 'from JanusAI...' statements
                    if old_prefix_from in new_content:
                        count = new_content.count(old_prefix_from)
                        new_content = new_content.replace(old_prefix_from, new_prefix_from)
                        replacements_in_file += count

                    # Fix 'import JanusAI...' statements
                    if old_prefix_import in new_content:
                        count = new_content.count(old_prefix_import)
                        new_content = new_content.replace(old_prefix_import, new_prefix_import)
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
        print("   No incorrect 'JanusAI' imports were found. Your project is already up to date!")
    
    if needs_rename:
        print("\n" + "!"*70)
        print("‼️ IMPORTANT NEXT STEP ‼️")
        print("   The script has fixed the imports inside the 'JanusAI' directory.")
        print("   You must now rename the directory for the imports to work.")
        print("   From your project root, please run this command:")
        print("\n      mv JanusAI janus\n")
        print("!"*70)
    
    print("="*70)


if __name__ == "__main__":
    fix_imports_in_project()
