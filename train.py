import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from load_data import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from model import create_model
from load_data import load_and_preprocess_data

# Define paths
train_images ,train_labels= load_and_preprocess_data(r'C:\Users\User\Downloads\lab\lab\anotated\images\train_images',r'C:\Users\User\Downloads\lab\lab\anotated\labels\train_labels',46)
# train_labels = load_and_preprocess_data()
val_images,val_labels = load_and_preprocess_data(r'C:\Users\User\Downloads\lab\lab\anotated\images\val_images',r'C:\Users\User\Downloads\lab\lab\anotated\labels\val_labels',46)
# val_labels = load_and_preprocess_data()


# Create the model
model = create_model(input_shape=(224, 224, 3), num_classes=46)  # Adjust as per your classes

# Set callbacks for early stopping and reducing learning rate
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

# Train the model
history = model.fit(
    train_images, train_labels, 
    epochs=6, 
    validation_data=(val_images, val_labels),
    callbacks=callbacks
)




# Plot training & validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Save the model
model.save('trained_model.h5')