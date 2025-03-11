import os
import pickle
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Collect image patch paths from multiple directories.')
    parser.add_argument('--input_dirs', nargs='+', required=True, 
                      help='List of input directories to search for patches')
    parser.add_argument('--output_file', required=True,
                      help='Path to save the output pickle file')
    parser.add_argument('--valid_folders_file', default=None,
                      help='Text file containing list of valid folders (optional)')
    parser.add_argument('--extension', default='.jpg',
                      help='File extension to filter (default: .jpg)')
    parser.add_argument('--exclude_extensions', nargs='+', default=['.h5'],
                      help='File extensions to exclude from folders (default: .h5)')
    
    args = parser.parse_args()
    
    # Load valid folders if provided
    valid_folders = None
    if args.valid_folders_file and os.path.exists(args.valid_folders_file):
        with open(args.valid_folders_file, 'r') as file:
            valid_folders = file.read().splitlines()
    
    data = []
    count = 0
    
    # Process each input directory
    for dir_index, input_dir in enumerate(args.input_dirs, 1):
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist. Skipping.")
            continue
            
        for folder in tqdm(os.listdir(input_dir), desc=f'Processing {input_dir}'):
            folder_path = os.path.join(input_dir, folder)
            
            # Skip if folder is actually a file with excluded extension
            if any(folder.endswith(ext) for ext in args.exclude_extensions):
                continue
                
            # Skip if it's not a directory
            if not os.path.isdir(folder_path):
                continue
                
            # Skip if directory is empty
            if len(os.listdir(folder_path)) == 0:
                continue
                
            # Skip if using valid_folders filter and this folder is not in the list
            if valid_folders is not None and folder not in valid_folders:
                continue
                
            count += 1
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(args.extension):
                    # Store relative path and directory index
                    relative_path = file_path.split(input_dir)[-1].lstrip(os.sep)
                    data.append((relative_path, dir_index))
    
    # Validate data entries
    print('Final Checking...')
    remove_list = []
    for i in tqdm(range(len(data))):
        if not data[i][0].endswith(args.extension):
            remove_list.append(i)
            
    for i in sorted(remove_list, reverse=True):
        data.pop(i)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save data to pickle file
    with open(args.output_file, 'wb') as file:
        pickle.dump(data, file)
    
    print(f"Collected: {len(data)} patches from {count} WSI directories.")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()