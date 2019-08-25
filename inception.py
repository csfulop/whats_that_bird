"""
Inference with Inception model

Usage:
    python inception.py <image>...
"""

#######################################

import numpy as np
from keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 299


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return (output)


#######################################

import sys
from keras.applications import InceptionV3

my_model = InceptionV3(weights='imagenet')
img_paths = sys.argv[1:]
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

#######################################

import pprint
from keras.applications.inception_v3 import decode_predictions

print('Predicted:')
pprint.pprint(
    list(zip(
        img_paths,
        decode_predictions(preds, top=3)
    ))
)
