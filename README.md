# What's that bird

Recognize birds with ML

## Pre-requisites

Install Anaconda 3: http://docs.anaconda.com/anaconda/install/linux/

With anaconda install TensorFlow and Keras:
```
conda activate
conda install tensorflow
conda install keras
```

## How to use

### `resnet.py`

This script inference with the original ResNet50 model with imagenet weights. Nothing bird specific.

Inside anaconda virtualenv:
```
python resnet.py <image>...
```
