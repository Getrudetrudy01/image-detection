import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model
model = tf.keras.models.load_model('item_detector_model.keras')

def predict(image_path):
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file at {image_path} does not exist.")
    
    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not loaded from path: {image_path}")
    
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    prediction = model.predict(image_input)
    
    # Post-process to extract bounding boxes
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return class_id, confidence

# Example prediction (provide a full path to a specific image file)
image_path = r'C:\Users\User\Downloads\lab\lab\anotated\images\val_images\IMG-20240724-WA0190.jpg'
class_id, confidence = predict(image_path)

class_names = ['dog', 'person', 'cat', 'tv', 'car', 'meatballs', 'marinara sauce', 'tomato soup', 'chicken noodle soup', 'french onion soup', 'chicken breast', 'ribs', 'pulled pork', 'hamburger', 'cavity', 'LUX soap', 'Sunlight', 'BF Soap', 'Guardian soap', 'Sunlight liquid soap', 'Geisha soap', 'CloseUp', 'Pepsodent', 'Yazz', 'Medisoft', 'Key soap long bar', 'Alive soap', 'Rexona', 'Key soap', 'Insecticide spray', 'lifebuoy', 'Sachet Pepsodent', 'P', 'Match box', 'Tin tomatoes', 'Pampers', 'Madar soap', 'Body Cream', 'Liquid soap', 'Deodorant', 'Detergent', 'Detol', 'ToothBrush', 'Anapuna salt', 'Angola', 'Colgate']
predicted_class = class_names[class_id]
print('Predicted class:', predicted_class, 'with confidence:', confidence)
