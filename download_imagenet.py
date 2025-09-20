import os
import json
import sys

def create_datalist():
    """
    Scans the ImageNet training directory, reads a class index JSON that maps
    folder names to class labels, and creates a 'train.txt' file.
    """
    # --- Configuration ---
    # Define the base directory where 'data' and the JSON file are located.
    base_dir = "."
    
    # Path to the root of the ImageNet training set (contains class folders like 'n01440764').
    image_root_dir = os.path.join(base_dir, "data", "imagenet", "train")
    
    # Path to the JSON file mapping class folder names to integer labels.
    json_index_path = os.path.join(base_dir, "imagenet_class_index.json")
    
    # Path for the output file.
    output_file_path = os.path.join(base_dir, "data", "imagenet", "train.txt")
    # --- End Configuration ---

    # --- 1. Validation ---
    print("Validating paths...")
    if not os.path.isdir(image_root_dir):
        print(f"Error: Image directory not found at '{image_root_dir}'")
        print("Please make sure the ImageNet 'train' directory is correctly placed.")
        sys.exit(1)
        
    if not os.path.exists(json_index_path):
        print(f"Error: JSON class index not found at '{json_index_path}'")
        sys.exit(1)
    
    print("Paths validated successfully.")

    # --- 2. Load the class index mapping ---
    print(f"Loading class index from '{json_index_path}'...")
    try:
        with open(json_index_path, 'r') as f:
            # The JSON directly maps folder names to integer labels, so we can use it as is.
            folder_to_label = json.load(f)
        print("Successfully loaded folder-to-label mapping.")
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse the JSON file. Details: {e}")
        sys.exit(1)


    # --- 3. Walk through the directory and generate file list ---
    print(f"Scanning for JPEG files in '{image_root_dir}'...")
    output_lines = []
    image_count = 0
    skipped_folders = 0

    # Sort the directories for a consistent output file order
    class_folders = sorted(os.listdir(image_root_dir))

    for class_folder in class_folders:
        class_folder_path = os.path.join(image_root_dir, class_folder)
        if not os.path.isdir(class_folder_path):
            continue

        # Look up the label for this folder
        if class_folder not in folder_to_label:
            print(f"  - Warning: Folder '{class_folder}' not found in JSON index. Skipping.")
            skipped_folders += 1
            continue
        
        label = folder_to_label[class_folder]

        # Sort filenames for a consistent output order
        filenames = sorted(os.listdir(class_folder_path))
        for filename in filenames:
            # Check for JPEG extension (case-insensitive)
            if filename.lower().endswith('.jpeg'):
                # Construct the relative path using forward slashes for compatibility
                relative_path = f"{class_folder}/{filename}"
                
                # Format the line as: path/to/image.JPEG label
                output_lines.append(f"{relative_path} {label}\n")
                image_count += 1
                
                if image_count % 50000 == 0:
                    print(f"  ... processed {image_count} images")

    print(f"\nScan complete. Found {image_count} images.")
    if skipped_folders > 0:
        print(f"Skipped {skipped_folders} folders that were not in the JSON index.")


    # --- 4. Write the output file ---
    print(f"Writing data list to '{output_file_path}'...")
    try:
        with open(output_file_path, 'w') as f:
            f.writelines(output_lines)
        print("Successfully created train.txt!")
    except IOError as e:
        print(f"Error: Could not write to file '{output_file_path}'.\nDetails: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_datalist()
    

# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path
# import os
# import tarfile
# import time

# def extract_with_parallel_io(archive_path, split, base_dir, num_workers=4):
#     """
#     Alternative: Parallel I/O version for maximum speed on SSDs.
#     This version uses the ThreadPoolExecutor's internal queue to avoid deadlocks.
#     It also deletes the archive file after a successful full extraction.
#     """
#     print(f"Extracting {archive_path} with {num_workers} workers...")
#     start_time = time.time()
    
#     base_path = Path(base_dir)
#     # Ensure the base split directory exists to avoid race conditions later
#     (base_path / split).mkdir(parents=True, exist_ok=True)
    
#     extracted_count = 0

#     def _write_file(data, target_path):
#         """Worker function: ensures dir exists and writes data."""
#         try:
#             # The worker is responsible for its own directory creation.
#             target_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(target_path, 'wb') as f:
#                 f.write(data)
#             return True
#         except Exception as e:
#             print(f"Worker error writing to {target_path}: {e}")
#             return False

#     try:
#         with tarfile.open(archive_path, 'r:gz') as tar:
#             # Get a list of all JPEG files to be extracted to check for completeness later
#             jpeg_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.JPEG')]
#             total_files_to_extract = len(jpeg_members)

#             if total_files_to_extract == 0:
#                 print(f"No JPEG files found in {archive_path}. Skipping extraction.")
#             else:
#                 with ThreadPoolExecutor(max_workers=num_workers) as executor:
#                     futures = []
#                     for member in jpeg_members:
#                         filename = os.path.basename(member.name)
                        
#                         if split == "test":
#                             target_path = base_path / split / filename
#                         else:
#                             synset_id = filename.split('_', 1)[0]
#                             target_path = base_path / split / synset_id / filename
                        
#                         # Read file data in the main thread
#                         file_data = tar.extractfile(member).read()
                        
#                         # Submit the write task to the thread pool
#                         future = executor.submit(_write_file, file_data, target_path)
#                         futures.append(future)

#                 # Wait for all futures to complete and count successful results
#                 for future in futures:
#                     if future.result():
#                         extracted_count += 1
                        
#         elapsed = time.time() - start_time
#         rate = extracted_count / elapsed if elapsed > 0 else 0
#         print(f"Parallel extraction of {archive_path} completed: {extracted_count} images in {elapsed:.1f}s ({rate:.1f} img/sec)")

#         if total_files_to_extract > 0 and extracted_count == total_files_to_extract:
#             try:
#                 os.remove(archive_path)
#                 print(f"Successfully extracted all {extracted_count} files. DELETED archive: {archive_path}")
#             except OSError as e:
#                 print(f"Error deleting file {archive_path}: {e}")
#         elif total_files_to_extract > 0:
#             print(f"WARNING: Mismatch in file counts. Extracted {extracted_count}/{total_files_to_extract}. Archive NOT DELETED: {archive_path}")
#         else:
#             print(f"Archive {archive_path} contained no JPEG files to extract. Archive NOT DELETED.")
    
#     except tarfile.ReadError as e:
#         print(f"Error reading tar file {archive_path}: {e}. Archive will not be deleted.")
#     except Exception as e:
#         print(f"An unexpected error occurred during extraction of {archive_path}: {e}. Archive will not be deleted.")
        
#     return extracted_count

# # Your main function remains the same.
# def main():
#     """Main function with usage examples."""
#     archive_dir = "/workspace/imagenet_cache/datasets--ILSVRC--imagenet-1k/snapshots/4603483700ee984ea9debe3ddbfdeae86f6489eb/data/"
#     archives = [
#         (archive_dir + "train_images_0.tar.gz", "train"),
#         (archive_dir + "train_images_1.tar.gz", "train"),
#         (archive_dir + "train_images_2.tar.gz", "train"),
#         (archive_dir + "train_images_3.tar.gz", "train"),
#         (archive_dir + "train_images_4.tar.gz", "train"),
#         # (archive_dir + "val_images.tar.gz", "val"),
#         # (archive_dir + "test_images.tar.gz", "test"),
#     ]
    
#     total_extracted = 0
#     total_start_time = time.time()
    
#     for archive, split in archives:
#         if not os.path.exists(archive):
#             print(f"Archive not found: {archive}. Skipping.")
#             continue
#         count = extract_with_parallel_io(archive, split, "data/imagenet", num_workers=8)
#         total_extracted += count

#     total_elapsed = time.time() - total_start_time
#     avg_rate = total_extracted / total_elapsed if total_elapsed > 0 else 0
    
#     print(f"\n=== SUMMARY ===")
#     print(f"Total images extracted: {total_extracted:,}")
#     print(f"Total time: {total_elapsed:.1f} seconds")
#     print(f"Average rate: {avg_rate:.1f} images/second")

# if __name__ == "__main__":
#     print("Optimized ImageNet Extractor")
#     main()


# import os
# from pathlib import Path

# def find_unlinked_blob_files(blobs_dir, data_dir):
#     """
#     Finds files in a 'blobs' directory that are not pointed to by any
#     symbolic link in a 'data' (snapshots) directory.

#     Args:
#         blobs_dir (str): The path to the directory containing the actual files (blobs).
#         data_dir (str): The path to the directory containing the symbolic links.

#     Returns:
#         list: A sorted list of unlinked file paths.
#     """
#     print("Verifying directory paths...")
#     try:
#         blobs_path = Path(blobs_dir).resolve(strict=True)
#         data_path = Path(data_dir).resolve(strict=True)
#     except FileNotFoundError as e:
#         print(f"Error: A directory does not exist. {e}")
#         return None

#     print("Step 1: Gathering all file paths from the blobs directory...")
#     # Get the real, absolute path for every file in the blobs directory.
#     # Using a set for fast lookups and differencing.
#     blob_files = {p.resolve() for p in blobs_path.glob('**/*') if p.is_file()}
#     print(f"Found {len(blob_files):,} actual files in {blobs_path}")

#     print("\nStep 2: Resolving all symlink targets from the data directory...")
#     # Get the real, absolute path for the TARGET of each symlink.
#     linked_files = set()
#     for p in data_path.glob('**/*'):
#         if p.is_symlink():
#             try:
#                 # .resolve() gets the final, canonical path of the link's target
#                 linked_files.add(p.resolve())
#             except FileNotFoundError:
#                 # This is a broken symlink, we can ignore it or report it
#                 # print(f"Warning: Found broken symlink, skipping: {p}")
#                 pass
#     print(f"Found {len(linked_files):,} unique files targeted by symlinks in {data_path}")

#     print("\nStep 3: Finding files in blobs that are not linked...")
#     # Use set difference to find files in blob_files that are not in linked_files
#     unlinked_files = blob_files - linked_files

#     return sorted(list(unlinked_files))

# if __name__ == "__main__":
#     # --- CONFIGURE YOUR PATHS HERE ---
#     BLOBS_DIRECTORY = "/workspace/imagenet_cache/datasets--ILSVRC--imagenet-1k/blobs"
#     DATA_DIRECTORY = "/workspace/imagenet_cache/datasets--ILSVRC--imagenet-1k/snapshots/4603483700ee984ea9debe3ddbfdeae86f6489eb/data"
    
#     unlinked = find_unlinked_blob_files(BLOBS_DIRECTORY, DATA_DIRECTORY)
    
#     if unlinked is not None:
#         if not unlinked:
#             print("\n✅ Success! All files in the blobs directory are correctly linked.")
#         else:
#             print(f"\n🚨 Found {len(unlinked)} unlinked files in '{BLOBS_DIRECTORY}':")
#             for file_path in unlinked:
#                 print(file_path)