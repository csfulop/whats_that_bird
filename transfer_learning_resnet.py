"""
Transfer learning with Keras and ResNet.
Train the model to recognize birds.

Based on:
    https://www.kaggle.com/dansbecker/transfer-learning
    https://www.kaggle.com/dansbecker/data-augmentation
"""

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip=True,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             rotation_range=15,
                                             shear_range=0.1,
                                             zoom_range=0.2)

train_generator = data_generator_with_aug.flow_from_directory(
    './bird_photos/train',
    target_size=(image_size, image_size),
    class_mode='categorical')

validation_generator = data_generator_with_aug.flow_from_directory(
    './bird_photos/test',
    target_size=(image_size, image_size),
    class_mode='categorical')

assert train_generator.num_classes == validation_generator.num_classes, 'training and validation classes must match'

#######################################

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

num_classes = train_generator.num_classes

bird_model = Sequential()
bird_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
bird_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
bird_model.layers[0].trainable = False

#######################################

bird_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#######################################

bird_model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=1)
