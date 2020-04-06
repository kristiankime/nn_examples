# Based on
# https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features

# Multiple features
# https://www.quora.com/How-can-I-make-an-RNN-model-in-Python-which-has-multiple-features

# variable length
# https://easy-tensorflow.com/tf-tutorials/recurrent-neural-networks/many-to-one-with-variable-sequence-length
# https://stackoverflow.com/questions/34670112/how-to-deal-with-batches-with-variable-length-sequences-in-tensorflow
# https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

# Misc
# https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/09-RNN-Sentiment-Analysis

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
training_size = 8000
eval_size = 1000

max_timesteps = 8
feature_num = 3


# =========== sample data generation functions
def series(timesteps, feature_num):
    return np.random.random((timesteps, feature_num))


def labels_from_data(data):
    return data # the identity function


# =========== Inputs
# first we create the dataset
train_data = np.array([ series(max_timesteps, feature_num) for i in range(0, training_size) ])
train_labels = labels_from_data(train_data)

# =========== Model Structure
# https://keras.io/
model = tf.keras.Sequential()

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
model.add(layers.Input(shape=(max_timesteps, feature_num)))

# This RNN will return timesteps with 3 features each.
# Because return_sequences=False, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 3), which matches the shape of each data point in data_y above.
model.add(layers.LSTM(feature_num, activation=None, return_sequences=True))

# =========== Model Prep
# https://www.tensorflow.org/guide/keras/overview#train_and_evaluate
model.compile(loss=losses.MSE, optimizer=optimizers.SGD(), metrics=[metrics.MSE])

# =========== Model Training
model.fit(train_data, train_labels, epochs=50)

# =========== Model Evaluation
# eval data
eval_data = np.array([ series(max_timesteps, feature_num) for i in range(0, eval_size) ])
eval_labels = labels_from_data(eval_data)
# evaluation
loss_and_metrics = model.evaluate(eval_data, eval_labels, batch_size=128)
print(loss_and_metrics)

# =========== Prediction
predict_data = np.array([
    # Series 1
    [
        [.05, .25, .45], # Features at timestep 1
        [.15, .35, .55], # Features at timestep 2
        [.40, .50, .60], # Features at timestep 3
        [.25, .50, .75], # Features at timestep 4
        [.05, .25, .45], # Features at timestep 5
        [.15, .35, .55], # Features at timestep 6
        [.40, .50, .60], # Features at timestep 7
        [.25, .50, .75], # Features at timestep 8
    ],
    # Series 2
    [
        [.05, .25, .45], # Features at timestep 1
        [.15, .35, .55], # Features at timestep 2
        [.40, .50, .60], # Features at timestep 3
        [.25, .50, .75], # Features at timestep 4
        [.05, .25, .45], # Features at timestep 5
        [.15, .35, .55], # Features at timestep 6
        [.40, .50, .60], # Features at timestep 7
        [.25, .50, .75], # Features at timestep 8
    ]
])

predictions = model.predict(predict_data)
print(predictions)