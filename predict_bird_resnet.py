import json
import pprint
import sys

import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

#######################################

image_size = 224


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return (output)


#######################################

bird_model = load_model('birds_resnet50_model.h5')

with open('birds_resnet50_classes.json') as f:
    classes_dict = json.load(f)
    classes = [x[0] for x in sorted(classes_dict.items(), key=lambda x: x[1])]

img_paths = sys.argv[1:]
test_data = read_and_prep_images(img_paths)
preds = bird_model.predict(test_data)
print('Predicted:')
pprint.pprint(list(zip(img_paths, [list(zip(classes, x)) for x in preds])))
