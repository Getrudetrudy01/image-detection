import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Load data and model
from load_data import load_and_preprocess_data

val_images_dir = r'C:\Users\User\Downloads\lab\lab\anotated\images\val_images'
val_labels_dir = r'C:\Users\User\Downloads\lab\lab\anotated\labels\val_labels'
num_classes = 46

X_val, y_val = load_and_preprocess_data(val_images_dir, val_labels_dir, num_classes)
model = tf.keras.models.load_model('item_detector_model.h5')

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f'Test accuracy: {test_accuracy:.2f}')
