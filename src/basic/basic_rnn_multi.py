# https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np

# the input values are random so set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)

# =========== Overview

# =========== Inputs
# first we create the dataset

# The inputs to the model.
# We will create two data points, just for the example.
train_data = np.array([
    # Datapoint 1
    [
        # Input features at timestep 1
        [1, 2, 3],
        # Input features at timestep 2
        [4, 5, 6]
    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [7, 8, 9],
        # Features at timestep 2
        [10, 11, 12]
    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
train_labels = np.array([
    # Datapoint 1
    # Target features at timestep 2
    [105, 106, 107, 108],
    # Datapoint 2
    # Target features at timestep 2
    [205, 206, 207, 208]
])


# =========== Model Structure
# https://keras.io/
model = tf.keras.Sequential()

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
model.add(layers.Input(shape=(2, 3)))

# This RNN will return timesteps with 4 features each.
# Because return_sequences=False, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 4), which matches the shape of each data point in data_y above.
model.add(layers.LSTM(4, return_sequences=False))

# =========== Model Prep
# https://www.tensorflow.org/guide/keras/overview#train_and_evaluate
model.compile(loss=losses.MSE, optimizer=optimizers.SGD(), metrics=[metrics.MSE])

# =========== Model Training
model.fit(train_data, train_labels)
