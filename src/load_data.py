import os
import shutil

def copy_files(src_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    copied_files = []

    # Check if processed_files.txt exists
    processed_files_path = "data/processed_files.txt"
    if os.path.exists(processed_files_path):
        # Read the list of processed files
        with open(processed_files_path, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    # Walk through the source directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Check if the file has been processed before
            if file not in processed_files:
                
                print(f"Processing '{file}'")

                # Construct source and destination file paths
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                # Copy the file to the destination directory
                shutil.copy(src_file, dest_file)
                copied_files.append(file)
                print(f"Copied '{src_file}' to '{dest_file}'")  

    # Write the list of copied files to processed_files.txt
    if len(copied_files) > 0:
        with open(processed_files_path, "a") as f:
            f.write("\n".join(copied_files) + "\n") 

if __name__ == "__main__":
    src_directory = "data/raw"
    dest_directory = "data/processed"
    copy_files(src_directory, dest_directory)
