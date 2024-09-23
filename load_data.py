import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
import keras
import np_utils
from keras._tf_keras.keras.utils import to_categorical

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

def load_yolo_data(image_dir, label_dir):
    """
    Load images and their YOLO formatted labels, convert labels to bounding box coordinates.
    """
    images = []
    labels = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: {image_path} not found.")
                continue
            h, w, _ = image.shape

            label_data = []
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    xmin = int((x_center - width / 2) * w)
                    ymin = int((y_center - height / 2) * h)
                    xmax = int((x_center + width / 2) * w)
                    ymax = int((y_center + height / 2) * h)
                    label_data.append([class_id, xmin, ymin, xmax, ymax])
            
            images.append(image)
            labels.append(label_data)

    return images, labels

def preprocess_images(images):
    """
    Resize and normalize images.
    """
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (224, 224))  # Resize for model input size
        img_normalized = img_resized / 255.0
        processed_images.append(img_normalized)
    return np.array(processed_images)

def augment_data(images, labels):
    """
    Augment data using various transformations, not adjusting bounding boxes here.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        image = image.reshape((1,) + image.shape)  # Add batch dimension for augmentation
        for _ in range(5):  # Generate 5 augmented images per input image
            for batch in datagen.flow(image, batch_size=1):
                augmented_images.append(batch[0])
                augmented_labels.append(label)  # Keep original labels, adjust if bounding box transformation needed
                break

    return np.array(augmented_images), augmented_labels

def load_and_preprocess_data(image_dir, label_dir, num_classes):
    """
    Load and preprocess data from directories, including augmentation.
    """
    images, labels = load_yolo_data(image_dir, label_dir)
    X = preprocess_images(images)
    # Assuming classification labels for simplification
    y = np.array([label[0][0] for label in labels])  # Taking class of the first bounding box
    y = to_categorical(y, num_classes)

    X_augmented, y_augmented = augment_data(X, labels)
    y_augmented = np.array([to_categorical(int(label[0][0]), num_classes) for label in y_augmented])

    return np.concatenate((X, X_augmented)), np.concatenate((y, y_augmented))
