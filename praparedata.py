import os
import shutil

# Paths
dataset_path = r'C:\Users\User\Downloads\lab\lab\anotated\images'  # Path to the images
yolo_labels_path = r'C:\Users\User\Downloads\lab\lab\anotated\labels'  # Path to the labels
output_images_path = r'C:\Users\User\Downloads\lab\lab\anotated\output\images'  # Output directory for images
output_labels_path = r'C:\Users\User\Downloads\lab\lab\anotated\output\labels'  # Output directory for labels

# Create output directories
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

def organize_data():
    for label_file in os.listdir(yolo_labels_path):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            src_image = os.path.join(dataset_path, image_file)
            if os.path.exists(src_image):
                dst_image = os.path.join(output_images_path, image_file)
                dst_label = os.path.join(output_labels_path, label_file)
                shutil.copy(src_image, dst_image)
                shutil.copy(os.path.join(yolo_labels_path, label_file), dst_label)
                print(f'Copied {image_file} and {label_file} to output directories')

organize_data()
print("Data organized successfully.")