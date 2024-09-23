import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from keras._tf_keras.keras.models import Model
# from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout


from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Flatten, Dense, Dropout , Conv2D, MaxPooling2D
from keras._tf_keras.keras.optimizers import Adam

import tensorflow as tf

def create_model(input_shape, num_classes):
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # Freeze the convolutional base

    x = base_model.output
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)  # Additional convolution layers
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
