# https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features

# https://easy-tensorflow.com/tf-tutorials/recurrent-neural-networks/many-to-one-with-variable-sequence-length
# https://stackoverflow.com/questions/34670112/how-to-deal-with-batches-with-variable-length-sequences-in-tensorflow
# https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras
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
training_size = 2000

max_timesteps = 2
feature_num = 3


# =========== Inputs
# first we create the dataset

def series(timesteps, feature_num):
    return np.random.random((timesteps, feature_num))

train_data = np.array([ series(max_timesteps, feature_num) for i in range(0, training_size) ])

# https://stackoverflow.com/questions/13567345/how-to-calculate-the-sum-of-all-columns-of-a-2d-numpy-array-efficiently
def sum_series_features(series):
    return series.sum(axis=0)
train_labels = np.array([sum_series_features(series) for series in train_data]) # the labels are the grouped sums

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
model.add(layers.LSTM(feature_num, return_sequences=False))

# =========== Model Prep
# https://www.tensorflow.org/guide/keras/overview#train_and_evaluate
model.compile(loss=losses.MSE, optimizer=optimizers.SGD(), metrics=[metrics.MSE])

# =========== Model Training
model.fit(train_data, train_labels, epochs=20)

# =========== Model Evaluation
# eval data
eval_data = np.array([ series(max_timesteps, feature_num) for i in range(0, training_size) ])
eval_labels = np.array([sum_series_features(series) for series in train_data])
# evaluation
loss_and_metrics = model.evaluate(eval_data, eval_labels, batch_size=128)
print(loss_and_metrics)

# =========== Prediction
# predict_data = np.array([
#     # Series 1
#     [
#         # Input features at timestep 1
#         [1, 2, 4],
#         # Input features at timestep 2
#         [1, 2, 4]
#     ],
#     # Series 2
#     [
#         # Features at timestep 1
#         [10, 20, 40],
#         # Features at timestep 2
#         [10, 20, 40]
#     ]
# ])
