# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Integrated Gradients
# 
# https://github.com/ankurtaly/Integrated-Gradients

# %%
# Imports
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

#from InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions_and_gradients, top_label_id_and_score
from IntegratedGradients.integrated_gradients import integrated_gradients, random_baseline_integrated_gradients
from VisualizationLibrary.visualization_lib import Visualize, show_pil_image, pil_image


# %%
# configuration for notebook
IN_COLAB = False
USE_GPU = IN_COLAB and False  # TPU and GPU only available in COLAB environment
USE_TPU = IN_COLAB and ((not USE_GPU) ^ False)  # XOR; either use GPU or TPU, cannot use both at the same time

BASE_PATH = '.'
LOG_DIR = os.path.join(BASE_PATH, 'logs')
USE_RGB = False

if USE_RGB:
    MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'face_keypoint_model_rgb')
    INPUT_CHANNEL_COUNT = 3  # test with gray-scale to rgb 
else:
    MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'face_keypoint_model')
    INPUT_CHANNEL_COUNT = 1  # due to gray-scale image we have only one color channel

# load pre-trained model or train new model
if os.path.exists(MODEL_OUTPUT_DIR):
    model = keras.models.load_model(MODEL_OUTPUT_DIR)
    
model.summary()


# %%
# load data
TEST_DATA_FILE = os.path.join(BASE_PATH, 'data', 'test.csv')

# read raw data from csv
test_data_raw = pd.read_csv(TEST_DATA_FILE)


# %%
def from_str_to_image(string_list):
    return np.array([np.array(row.split(), dtype=np.uint8) for row in string_list])


def normalize_image(data):
    return (data / 255.0)

def normalize_image_batch(data, width, height, channel_count):
    return (data / 255.0).reshape((-1, width, height, channel_count)) # additional batch layer

def predictions_and_gradients(images, target_keypoint_idx):
    if len(images.shape) != 4:
        images = tf.expand_dims(images, axis=0)
    with tf.GradientTape() as tape:  # used for automatic gradient computation
        tape.watch(images)
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        # training=False required, when directly using __call__ and having batchnorm, etc.
        logits = model(images, training=False)
        preds = logits[:, target_keypoint_idx]
    return preds, tape.gradient(preds, images)


# %%
# define test image
image_index = 994

# prepare test data for prediction
# convert image string to Numpy array
test_data = from_str_to_image(test_data_raw['Image'])
if USE_RGB:
    # re-create rgb channels from gray-scale
    test_data = np.stack((test_data,)*3, axis=-1)

# extract image dimensions
IMG_WIDTH = IMG_HEIGHT = np.sqrt(test_data.shape[1]).astype(np.uint8)

# show first test image
plt.imshow(test_data[image_index, :].reshape((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT)), cmap='gray')


# %%
# prepare test image
test_image = test_data[image_index,:].reshape((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
test_image = normalize_image(test_image)
test_image = tf.constant(test_image, dtype=tf.float32)


# %%
# Compute attributions based on just the gradients.
target_keypoint_idx = 0  # [0, 29] for 30 face keypoints

predictions, gradients = predictions_and_gradients(test_image, target_keypoint_idx)


# %%
# print('Gradients')
# show_pil_image(pil_image(Visualize(
#     gradients[0], test_image,
#     clip_above_percentile=99,
#     clip_below_percentile=0,
#     overlay=True)))


# %%
# Compute attributions based on the integrated gradients method.
attributions = random_baseline_integrated_gradients(
    test_image,
    target_keypoint_idx,
    predictions_and_gradients,
    steps=50,
    num_random_trials=10)


# %%
print(predictions)


# %%
print('Integrated Gradients')
show_pil_image(pil_image(Visualize(
    attributions, test_image,
    clip_above_percentile=99,
    clip_below_percentile=0,
    overlay=True,
    polarity='positive',
    plot_distribution=False)))
