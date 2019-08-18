# -*- coding: utf-8 -*-
"""
Inference with ResNet model

Based on:
    https://www.kaggle.com/dansbecker/tensorflow-programming
    https://keras.io/applications/#usage-examples-for-image-classification-models

Usage:
    python resnet.py <image>...
"""

#######################################

import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

#######################################

from tensorflow.python.keras.applications import ResNet50
import sys

my_model = ResNet50(weights='imagenet')
img_paths = sys.argv[1:]
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

#######################################

from keras.applications.resnet50 import decode_predictions
import pprint
print('Predicted:') 
pprint.pprint(
    list(zip(
        img_paths,
        decode_predictions(preds, top=3)
    ))
)
