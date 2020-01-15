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
# We're going to train a very basic NN to predict a predict a pattern where every group of 10 inputs gets turned into one number by summing
# so in the case we start with 100 inputs then the 1st label is the sum of inputs 1-10, the 2nd is the sum of 11-20 etc.

input_dim = 100
output_dim = 10
group_size = int(input_dim / output_dim)


# =========== Inputs
# first we create the dataset

# Create a function to sum the groups https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
sum_groups = lambda row: np.array([sum(group) for group in np.split(row, group_size)])

train_data = np.random.random((3000, input_dim))
train_labels = np.array([sum_groups(row) for row in train_data]) # the labels are the grouped sums

# =========== Model Structure
# https://keras.io/
model = tf.keras.Sequential()

model.add(layers.Dense(units=64, input_dim=input_dim))
model.add(layers.Dense(units=output_dim))

# =========== Model Prep
# https://www.tensorflow.org/guide/keras/overview#train_and_evaluate
model.compile(loss=losses.MSE, optimizer=optimizers.SGD(), metrics=[metrics.MSE])

# =========== Model Training
model.fit(train_data, train_labels, epochs=50, batch_size=32)

# =========== Model Evaluation
# eval data
eval_data = np.random.random((1000, input_dim))
eval_labels = np.array([sum_groups(row) for row in eval_data]) # the labels are the grouped sums

# evaluation
loss_and_metrics = model.evaluate(eval_data, eval_labels, batch_size=128)
print(loss_and_metrics)

# =========== Prediction
predict_data = np.array([
         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,]
        ])

predictions = model.predict(predict_data)
print(predictions)
