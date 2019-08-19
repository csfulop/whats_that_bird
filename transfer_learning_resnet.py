"""
Transfer learning with Keras and ResNet.
Train the model to recognize birds.

Based on:
    https://www.kaggle.com/dansbecker/transfer-learning
"""

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 5

bird_model = Sequential()
bird_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
bird_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
bird_model.layers[0].trainable = False

#######################################

bird_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#######################################

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        './bird_photos/train',
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        './bird_photos/test',
        target_size=(image_size, image_size),
        class_mode='categorical')

bird_model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=1)
