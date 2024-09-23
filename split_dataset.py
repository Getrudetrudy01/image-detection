import os
import shutil
from sklearn.model_selection import train_test_split

def setup_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def move_files(files, source_dir, target_dir):
    """Move specified files from source to target directory."""
    for file in files:
        src_file = os.path.join(source_dir, file)
        dst_file = os.path.join(target_dir, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            print(f"Warning: {src_file} does not exist and cannot be moved.")

def main():
    # Defining the paths
    images_dir = 'C:/Users/User/Downloads/lab/lab/anotated/images'
    labels_dir = 'C:/Users/User/Downloads/lab/lab/anotated/labels'
    train_images_dir = 'C:/Users/User/Downloads/lab/lab/anotated/images/train_images'
    val_images_dir = 'C:/Users/User/Downloads/lab/lab/anotated/images/val_images'
    train_labels_dir = 'C:/Users/User/Downloads/lab/lab/anotated/labels/train_labels'
    val_labels_dir = 'C:/Users/User/Downloads/lab/lab/anotated/labels/val_labels'

    # Ensure the output directories exist
    setup_directory(train_images_dir)
    setup_directory(val_images_dir)
    setup_directory(train_labels_dir)
    setup_directory(val_labels_dir)

    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    # Split into training and validation sets
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)  # 20% for validation

    # Move image files
    move_files(train_files, images_dir, train_images_dir)
    move_files(val_files, images_dir, val_images_dir)
    # Move corresponding label files
    move_files([f.replace('.jpg', '.txt') for f in train_files], labels_dir, train_labels_dir)
    move_files([f.replace('.jpg', '.txt') for f in val_files], labels_dir, val_labels_dir)

    print("Dataset split into training and validation sets and moved successfully.")

if __name__ == '__main__':
    main()
