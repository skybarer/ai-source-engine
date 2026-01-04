#!/usr/bin/env python3
"""
Fix Unicode characters in Python files for Windows compatibility
"""

import os
import re
from pathlib import Path

# Unicode replacements
REPLACEMENTS = {
    '✓': '[OK]',
    '✗': '[FAIL]',
    '✅': '[OK]',
    '❌': '[FAIL]',
    '🔥': '[HOT]',
    '📊': '[CHART]',
    '📝': '[NOTE]',
    '🎓': '[DEGREE]',
    '⚠️': '[WARN]',
    '❗': '[ERROR]',
    '🚀': '[ROCKET]',
    '📋': '[LIST]',
    '📚': '[BOOKS]',
    '🎯': '[TARGET]',
    '💾': '[SAVE]',
    '📂': '[FOLDER]',
}

def fix_file(filepath):
    """Fix Unicode characters in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for unicode_char, replacement in REPLACEMENTS.items():
            content = content.replace(unicode_char, replacement)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FIXED] {filepath}")
            return True
        else:
            print(f"[OK] {filepath}")
            return False
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return False

def main():
    """Fix all Python files"""
    root_dir = Path(__file__).parent
    py_files = root_dir.glob('*.py')
    
    print("Fixing Unicode characters in Python files...")
    print("=" * 70)
    
    fixed_count = 0
    for py_file in py_files:
        if py_file.name != 'fix_unicode.py':
            if fix_file(py_file):
                fixed_count += 1
    
    print("=" * 70)
    print(f"✓ Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
