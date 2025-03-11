#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path


def scan_wsi_files(directory, extensions=None, recursive=False):
    """
    Scan a directory for WSI files with specified extensions.
    
    Args:
        directory (str): Directory path to scan
        extensions (list): List of file extensions to include (e.g., ['.sdpc'])
        recursive (bool): Whether to scan subdirectories recursively
        
    Returns:
        list: List of dictionaries with id and original_path for each WSI file
    """
    if extensions is None:
        raise ValueError("Extensions must be specified")
    
    wsi_files = []
    file_id = 0
    
    # Convert extensions to lowercase for case-insensitive comparison
    extensions = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in extensions]
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Choose the appropriate walking method based on recursive flag
    if recursive:
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    else:
        # Only list files in the top directory
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f))]
    
    # Filter and process files
    for file_path in sorted(files):
        if Path(file_path).suffix.lower() in extensions:
            wsi_files.append({
                "id": file_id,
                "original_path": file_path
            })
            file_id += 1
    
    return wsi_files


def main():
    parser = argparse.ArgumentParser(description="Scan a directory for WSI files and output JSON")
    parser.add_argument("directory", help="Directory to scan for WSI files")
    parser.add_argument("-e", "--extensions", nargs="+", 
                        help="File extensions to include (default: .svs)")
    parser.add_argument("-r", "--recursive", action="store_true", 
                        help="Scan subdirectories recursively")
    parser.add_argument("-o", "--output", default="wsi_files.json",
                        help="Output JSON file (default: wsi_files.json)")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print the JSON output, costs more storage")
    
    args = parser.parse_args()
    
    # Ensure the directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return 1
    
    # Scan for WSI files
    wsi_files = scan_wsi_files(args.directory, args.extensions, args.recursive)
    
    # Prepare JSON structure
    output_data = {"images": wsi_files}
    
    # Write to file
    with open(args.output, 'w') as f:
        if args.pretty:
            json.dump(output_data, f, indent=4)
        else:
            json.dump(output_data, f)
    
    print(f"Found {len(wsi_files)} WSI files. Output saved to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())