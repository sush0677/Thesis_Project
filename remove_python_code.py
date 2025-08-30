#!/usr/bin/env python3
"""
Script to remove all Python code blocks from final report markdown files
and replace them with descriptive text.
"""

import os
import re
import glob

def remove_python_code_blocks(file_path):
    """Remove Python code blocks from a markdown file."""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match Python code blocks
    python_code_pattern = r'```python\n.*?\n```'
    
    # Find all Python code blocks
    matches = re.findall(python_code_pattern, content, re.DOTALL)
    
    if matches:
        print(f"Found {len(matches)} Python code blocks in {file_path}")
        
        # Replace each Python code block with a descriptive comment
        for i, match in enumerate(matches):
            replacement = f"[*Technical implementation details available in source code repository*]"
            content = content.replace(match, replacement, 1)
        
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Removed {len(matches)} Python code blocks from {file_path}")
    else:
        print(f"No Python code blocks found in {file_path}")

def main():
    # Process all final report markdown files
    report_dir = r"c:\Users\SushantPatil\OneDrive - Nathan & Nathan\Documents\GitHub\Thesis_Project\docs\final_report"
    
    # Find all markdown files in the final report directory
    md_files = glob.glob(os.path.join(report_dir, "*.md"))
    
    for file_path in md_files:
        remove_python_code_blocks(file_path)
    
    print("All Python code blocks have been removed from final report files.")

if __name__ == "__main__":
    main()
