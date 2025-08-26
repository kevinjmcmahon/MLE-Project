import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from scipy.io import loadmat # For reading MATLAB .mat files

# --- Configuration ---
# Set up logging to show progress and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define key paths. Using Path objects is a modern and robust way to handle file paths.
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
IMAGE_DIR = RAW_DATA_PATH / "Images"

# Use the official train/test lists provided by the Stanford Dogs dataset in .mat format
TRAIN_LIST_PATH = RAW_DATA_PATH / "train_list.mat"
TEST_LIST_PATH = RAW_DATA_PATH / "test_list.mat"

# We will create our validation set by splitting the official training set.
# A 20% split is a standard choice.
VALIDATION_SPLIT_RATIO = 0.2
RANDOM_SEED = 42 # Using a fixed seed for reproducibility is crucial!

# --- Main Script Logic ---

def copy_files(file_paths, destination_split_name):
    """
    Copies a list of image files to the appropriate destination folder.
    
    Args:
        file_paths (list): A list of relative paths to the image files.
        destination_split_name (str): The name of the split ('train', 'validation', or 'test').
    """
    logging.info(f"Copying {len(file_paths)} files to '{destination_split_name}' set...")
    
    for file_path_str in tqdm(file_paths, desc=f"Copying to {destination_split_name}"):
        # The file paths in the list are like 'n02085620-Chihuahua/n02085620_10074.jpg'
        breed_folder_name = Path(file_path_str).parent.name
        
        # Create the destination breed folder (e.g., data/processed/train/n02085620-Chihuahua)
        destination_breed_dir = PROCESSED_DATA_PATH / destination_split_name / breed_folder_name
        destination_breed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the full source and destination paths
        source_file = IMAGE_DIR / file_path_str
        destination_file = destination_breed_dir / Path(file_path_str).name
        
        # Copy the file, preserving metadata
        shutil.copy2(source_file, destination_file)

def main():
    """Main function to orchestrate the data preparation process."""
    
    # 1. Housekeeping: Clean up and create directory structure
    logging.info("Starting data preparation process.")
    if PROCESSED_DATA_PATH.exists():
        logging.warning(f"Output directory '{PROCESSED_DATA_PATH}' already exists. Removing it for a fresh start.")
        shutil.rmtree(PROCESSED_DATA_PATH)
    
    logging.info("Creating new processed data directory structure.")
    # This loop creates data/processed/train, data/processed/validation, and data/processed/test
    for split in ["train", "validation", "test"]:
        (PROCESSED_DATA_PATH / split).mkdir(parents=True)

    # 2. Read the official train and test lists from the .mat files
    logging.info("Reading official train/test lists from .mat files.")
    
    # loadmat returns a dictionary. The file paths are under the 'file_list' key.
    train_data = loadmat(TRAIN_LIST_PATH)
    test_data = loadmat(TEST_LIST_PATH)
    
    # The data is nested in a complex structure, so we need to flatten it.
    # This list comprehension extracts the string path from each item.
    original_train_files = [item[0][0] for item in train_data['file_list']]
    test_files = [item[0][0] for item in test_data['file_list']]
        
    # 3. Create a stratified split for our new train and validation sets
    # We need labels (the breed folder) to stratify correctly.
    train_labels = [Path(p).parent.name for p in original_train_files]
    
    logging.info(f"Splitting the {len(original_train_files)} original training images into new train and validation sets.")
    
    train_files, validation_files = train_test_split(
        original_train_files,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,      # Guarantees the same split every time
        stratify=train_labels         # Ensures breed proportions are the same in train and val sets
    )

    # 4. Copy all files to their new homes
    copy_files(train_files, "train")
    copy_files(validation_files, "validation")
    copy_files(test_files, "test")

    # 5. Final verification
    logging.info("Data preparation complete. Verifying file counts...")
    train_count = len(list(PROCESSED_DATA_PATH.glob('train/**/*.jpg')))
    val_count = len(list(PROCESSED_DATA_PATH.glob('validation/**/*.jpg')))
    test_count = len(list(PROCESSED_DATA_PATH.glob('test/**/*.jpg')))
    
    logging.info(f"Total training images: {train_count}")
    logging.info(f"Total validation images: {val_count}")
    logging.info(f"Total test images: {test_count}")
    logging.info(f"Total images: {train_count + val_count + test_count}")

if __name__ == "__main__":
    main()