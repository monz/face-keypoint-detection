# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Inspect Model with Integrated Gradient
# 
# https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
# 

# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras


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
    #MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'face_keypoint_model')
    MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, 'face_keypoint_model_no_batchnorm')
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


def normalize_image(data, width, height, channel_count):
    return (data / 255.0).reshape((width, height, channel_count)) # no batch layer


def normalize_image_batch(data, width, height, channel_count):
    return (data / 255.0).reshape((-1, width, height, channel_count)) # additional batch layer


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
# prediction test for RGB image variant
test_image = test_data[image_index, :]

print(test_image.shape)
normalized_img = normalize_image_batch(test_image, IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT)
print(normalized_img.shape)

#assert(False)
if USE_RGB:
    POINT_SCALE_FACTOR = 1 # used to be 96 (img_width or height)
else:
    POINT_SCALE_FACTOR = 1
points = model.predict(normalized_img)*POINT_SCALE_FACTOR


# show test image
plt.imshow(test_image.reshape((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT)), cmap='gray')

# show face keypoints on test image
for x, y in zip(points[0,::2], points[0,1::2]):
    print(x, y)
    plt.scatter(x, y, color='r', linewidth=1)


# %%
def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    
    return images


def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:  # used for automatic gradient computation
        tape.watch(images)
        #logits = model(images)*POINT_SCALE_FACTOR
        logits = model(images)
        #probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        preds = logits[:, target_class_idx]
        #preds = tf.nn.softmax(logits/96.0, axis=-1)[:, target_class_idx]
    return tape.gradient(preds, images)


def predictions_and_gradients(images, target_class_idx):
    if len(images.shape) != 4:
        images = tf.expand_dims(images, axis=0)
    with tf.GradientTape() as tape:  # used for automatic gradient computation
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return probs, tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


@tf.function
def integrated_gradients(baseline, image, target_class_idx, m_steps=50, batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    #for alpha in tf.range(0, len(alphas), batch_size):
    for alpha in range(0, m_steps+1, batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, m_steps+1)
        alpha_batch = alphas[from_:to]
        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)
        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(images=interpolated_path_input_batch, target_class_idx=target_class_idx)
        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def by_polarity(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise Exception("unimplemented")


def ComputeThresholdByTopPercentage(attributions,
                                    percentage=60):
    """Compute the threshold value that maps to the top percentage of values.
    This function takes the cumulative sum of attributions and computes the set
    of top attributions that contribute to the given percentage of the total sum.
    The lowest value of this given set is returned.
    Args:
        attributions: (numpy.array) The provided attributions.
        percentage: (float) Specified percentage by which to threshold.
        plot_distribution: (bool) If true, plots the distribution of attributions
          and indicates the threshold point by a vertical line.
    Returns:
        (float) The threshold value.
    Raises:
        ValueError: if percentage is not in [0, 100].
    """
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
  
    # For percentage equal to 100, this should in theory return the lowest
    # value as the threshold. However, due to precision errors in numpy's cumsum,
    # the last value won't sum to 100%. Thus, in this special case, we force the
    # threshold to equal the min value.
    if percentage == 100:
        return np.min(attributions)

    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)

    # Sort the attributions from largest to smallest.
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]

    # Compute a normalized cumulative sum, so that each attribution is mapped to
    # the percentage of the total sum that it and all values above it contribute.
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]

    return threshold


def LinearTransform(attributions,
                    clip_above_percentile=99.9,
                    clip_below_percentile=70.0,
                    low=0.2):
    """Transform the attributions by a linear function.
    Transform the attributions so that the specified percentage of top attribution
    values are mapped to a linear space between `low` and 1.0.
    Args:
        attributions: (numpy.array) The provided attributions.
        percentage: (float) The percentage of top attribution values.
        low: (float) The low end of the linear space.
    Returns:
        (numpy.array) The linearly transformed attributions.
    Raises:
        ValueError: if percentage is not in [0, 100].
  """
    if clip_above_percentile < 0 or clip_above_percentile > 100:
        raise ValueError('clip_above_percentile must be in [0, 100]')

    if clip_below_percentile < 0 or clip_below_percentile > 100:
        raise ValueError('clip_below_percentile must be in [0, 100]')

    if low < 0 or low > 1:
        raise ValueError('low must be in [0, 1]')

    m = ComputeThresholdByTopPercentage(attributions,
                                      percentage=100-clip_above_percentile)
    e = ComputeThresholdByTopPercentage(attributions,
                                      percentage=100-clip_below_percentile)

    # Transform the attributions by a linear function f(x) = a*x + b such that
    # f(m) = 1.0 and f(e) = low. Derivation:
    #   a*m + b = 1, a*e + b = low  ==>  a = (1 - low) / (m - e)
    #                               ==>  b = low - (1 - low) * e / (m - e)
    #                               ==>  f(x) = (1 - low) (x - e) / (m - e) + low
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low

    # Recover the original sign of the attributions.
    transformed *= np.sign(attributions)

    # Map values below low to 0.
    transformed *= (transformed >= low)

    # Clip values above and below.
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed

G = [0, 255, 0]
R = [255, 0, 0]
def img_attributions(attributions,
                     overlay=False,
                     polarity='positive',
                     clip_above_percentile=99.9,
                     clip_below_percentile=0,
                     positive_channel=G,
                     negative_channel=R):
    

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    #attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    if polarity == 'both':
        attributions_positive = img_attributions(attributions,
                                                      overlay=False,
                                                      polarity='positive')
        
        attributions_negative = img_attributions(attributions,
                                                      overlay=False,
                                                      polarity='negative')
        
        attributions = attributions_positive + attributions_negative
        
        return attributions
    elif polarity == 'positive':
        attributions = by_polarity(attributions, polarity=polarity)
        channel = positive_channel
    elif polarity == 'negative':
        attributions = by_polarity(attributions, polarity=polarity)
        attributions = np.abs(attributions)
        channel = negative_channel
    else:
        raise Exception("unimplemented")
    
    attributions = np.average(attributions, axis=-1)
    
    attributions = LinearTransform(attributions,
                                 clip_above_percentile, clip_below_percentile,
                                 0.0)

    # Convert to RGB space
    attributions = np.expand_dims(attributions, 2) * channel

    # scale the attributes to [0,255]
    return np.clip(attributions, 0, 255).astype(int)


def plot_img_attributions(attributions,
                          baseline,
                          image,
                          keypoint_x,
                          keypoint_y,
                          cmap=None,
                          overlay_alpha=0.4):

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image, cmap='gray')
    axs[0, 1].scatter(keypoint_x, keypoint_y, color='r', linewidth=1)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attributions, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attributions, cmap=cmap)
    axs[1, 1].imshow(image, cmap='gray', alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()


# %%
# create alphas for interpolation
m_steps = 50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

# create baseline image, which is all black
baseline = tf.zeros(shape=(IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
# prepare test image
test_image = test_data[image_index,:].reshape((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
test_image = normalize_image(test_image, IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT)
test_image = tf.constant(test_image, dtype=tf.float32)  # convert to tensor

# interpolate images using alphas
interpolated_images = interpolate_images(baseline=baseline, image=test_image, alphas=alphas)

# plot interpolated example images
fig = plt.figure(figsize=(20, 20))
i = 0
for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
    i += 1
    plt.subplot(1, len(alphas[0::10]), i)
    plt.title(f'alpha: {alpha:.1f}')
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout();


# %%
preds = model(interpolated_images)
preds_some_keypoint_x = preds[:, 0]

plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, preds_some_keypoint_x)
ax1.set_title('Target keypoint coordinate predicted over alpha')
ax1.set_ylabel('model keypoint coordinate prediction')
ax1.set_xlabel('alpha')
#ax1.set_ylim([0, 1])

ax2 = plt.subplot(1, 2, 2)
# Average across interpolation steps
path_gradients = compute_gradients(images=interpolated_images, target_class_idx=0)
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
# Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1]);


# %%
# prepare test image
test_image = test_data[image_index,:].reshape((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
test_image = normalize_image(test_image, IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT)
test_image = tf.constant(test_image, dtype=tf.float32)

#baseline = tf.random.normal(shape=(IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
#baseline = tf.random.uniform(shape=(IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))
baseline = tf.zeros(shape=(IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNEL_COUNT))

# compute predictions for plot
predictions = model(tf.expand_dims(test_image, 0))

M_STEPS=50
TARGET_CLASS_IDX=0
COMBINED = True
POLARITY = 'positive'  # options are, 'both', 'positive', 'negative'
if COMBINED:
    for i in range(0, 30, 2):
        attributions_x = integrated_gradients(baseline=baseline,
                                            image=test_image,
                                            target_class_idx=i,
                                            m_steps=M_STEPS)
        
        attributions_y = integrated_gradients(baseline=baseline,
                                            image=test_image,
                                            target_class_idx=i+1,
                                            m_steps=M_STEPS)

        image_attributions_x = img_attributions(attributions_x,
                                              polarity=POLARITY)
        image_attributions_y = img_attributions(attributions_y,
                                              polarity=POLARITY)

        image_attributions = np.clip(image_attributions_x + image_attributions_y, 0, 255)
        
        keypoint_x = predictions[:, i]
        keypoint_y = predictions[:, i+1]
        plot_img_attributions(image_attributions,
                              baseline=baseline,
                              image=test_image,
                              keypoint_x=keypoint_x,
                              keypoint_y=keypoint_y,)
else:
    attributions = integrated_gradients(baseline=baseline,
                                        image=test_image,
                                        target_class_idx=TARGET_CLASS_IDX,
                                        m_steps=M_STEPS)
    
    image_attributions = img_attributions(attributions,
                                          polarity=POLARITY)
    
    
    keypoint_x = predictions[:, TARGET_CLASS_IDX]
    keypoint_y = predictions[:, TARGET_CLASS_IDX+1]
    
    plot_img_attributions(image_attributions,
                          baseline=baseline,
                          image=test_image,
                          keypoint_x=keypoint_x,
                          keypoint_y=keypoint_y,)
