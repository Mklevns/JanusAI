#!/usr/bin/env python3
"""
Quick Syntax Fix for pipeline.py
==============================

Specifically fixes the indentation error on line 116 of pipeline.py

Usage:
    python quick_syntax_fix.py
"""

import sys
from pathlib import Path


def fix_pipeline_syntax():
    """Fix the specific syntax error in pipeline.py."""
    pipeline_file = Path("src/janus/integration/pipeline.py")
    
    if not pipeline_file.exists():
        print(f"❌ {pipeline_file} not found")
        return False
    
    try:
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📄 Checking {len(lines)} lines in pipeline.py...")
        
        # Check line 116 specifically (index 115)
        if len(lines) >= 116:
            line_116 = lines[115]
            print(f"Line 116: '{line_116.rstrip()}'")
            
            # Check for common indentation issues
            fixed = False
            
            # Remove any trailing whitespace
            lines = [line.rstrip() + '\n' if line.strip() else line for line in lines]
            
            # Fix mixed tabs/spaces
            for i, line in enumerate(lines):
                if '\t' in line:
                    lines[i] = line.replace('\t', '    ')
                    print(f"Fixed tabs on line {i+1}")
                    fixed = True
            
            # Check for indentation consistency around line 116
            for i in range(max(0, 110), min(len(lines), 125)):
                line = lines[i]
                if line.strip() and not line.strip().startswith('#'):
                    # Count leading spaces
                    leading_spaces = len(line) - len(line.lstrip())
                    
                    # Ensure indentation is multiple of 4
                    if leading_spaces % 4 != 0 and leading_spaces > 0:
                        correct_spaces = ((leading_spaces + 2) // 4) * 4
                        lines[i] = ' ' * correct_spaces + line.lstrip()
                        print(f"Fixed indentation on line {i+1}: {leading_spaces} → {correct_spaces} spaces")
                        fixed = True
            
            # Look for specific problematic patterns
            for i in range(len(lines)):
                line = lines[i]
                
                # Fix common Python syntax issues
                if line.strip().startswith('except ImportError as e:') and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.strip() and not next_line.startswith('    '):
                        # Next line after except should be indented
                        lines[i + 1] = '    ' + next_line.lstrip()
                        print(f"Fixed except block indentation on line {i+2}")
                        fixed = True
                
                # Fix class method indentation
                if line.strip().startswith('def ') and not line.startswith('    def '):
                    if i > 0 and ('class ' in lines[i-1] or lines[i-1].strip().endswith(':')):
                        lines[i] = '    ' + line.lstrip()
                        print(f"Fixed method indentation on line {i+1}")
                        fixed = True
            
            # Try to compile to catch syntax errors
            content = ''.join(lines)
            try:
                compile(content, str(pipeline_file), 'exec')
                print("✅ Code compiles successfully!")
                
                if fixed:
                    # Write the fixed version
                    with open(pipeline_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print("✅ Fixed and saved pipeline.py")
                
                return True
                
            except SyntaxError as se:
                print(f"❌ Still has syntax error: {se}")
                print(f"   Error on line {se.lineno}: {se.text}")
                
                # Try to fix the specific error
                if se.lineno and se.lineno <= len(lines):
                    error_line = lines[se.lineno - 1]
                    print(f"   Problematic line: '{error_line.rstrip()}'")
                    
                    # Common fixes for specific syntax errors
                    if "unindent does not match any outer indentation level" in str(se):
                        # Find the correct indentation level
                        for j in range(se.lineno - 2, -1, -1):
                            if lines[j].strip() and not lines[j].strip().startswith('#'):
                                prev_indent = len(lines[j]) - len(lines[j].lstrip())
                                break
                        else:
                            prev_indent = 0
                        
                        # Fix the indentation
                        lines[se.lineno - 1] = ' ' * prev_indent + error_line.lstrip()
                        print(f"   Fixed indentation to {prev_indent} spaces")
                        
                        # Write and test again
                        with open(pipeline_file, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        
                        # Test compilation again
                        content = ''.join(lines)
                        try:
                            compile(content, str(pipeline_file), 'exec')
                            print("✅ Fixed! Code now compiles successfully!")
                            return True
                        except SyntaxError:
                            print("❌ Still has syntax errors")
                            return False
                
                return False
        
        else:
            print("❌ File has fewer than 116 lines")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("🔧 Quick Syntax Fix for pipeline.py")
    print("=" * 40)
    
    if fix_pipeline_syntax():
        print("\n🎉 Success! Try running the validation test again:")
        print("   python simple_validation_test.py")
    else:
        print("\n❌ Could not fix automatically. You may need to:")
        print("1. Check src/janus/integration/pipeline.py manually")
        print("2. Look for indentation issues around line 116")
        print("3. Ensure consistent use of spaces (not tabs)")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
