# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras

from numpy import array
from tensorflow.keras.models import Model

from logs import stdout_add_file, stdout_reset
from util import group_snapshots, read_numpy_3d_array_from_txt
from models import lstm_autoencoder
from data import split_snapshot_history_single, split_snapshot_history

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
# user_size = 500 # 3285
full_history_length = 243
model_history_length = 13 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
feature_num = 29 # <correct or not> + <28 features>

lstm_layer_size = 100
epochs = 240

# Get the model and switch to using the LSTM Layer as output
model_dir = os.path.join('runs', f'run_t{model_history_length}_l{lstm_layer_size}_e{epochs}')
embedding_model = keras.models.load_model(os.path.join(model_dir, f'model.h5'))
# connect the encoder LSTM as the output layer
embedding_model = Model(inputs=embedding_model.inputs, outputs=embedding_model.layers[0].output)


answer_snapshots = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_train_l{full_history_length}_10p.txt'))


(snapshots_embedded, snapshots_labels) = split_snapshot_history(embedding_model, answer_snapshots, model_history_length)

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()


# snapshot = answer_snapshots[100:101][0]
#
# (hist, final, label) = split_snapshot_history_single(snapshot, model_history_length)
#
# embeddings = model.predict(array(hist))
# embeddings_flat = embeddings.flatten()
# embedded_history = np.append(embeddings_flat, final)
#


# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/keras/classification?hl=nb

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(1928,)),  # input shape required
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2) # Binary output
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
train_inputs = snapshots_embedded
train_labels = snapshots_labels
model.fit(train_inputs, train_labels, epochs=10)


test_inputs = snapshots_embedded
test_labels = snapshots_labels
test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_inputs)
predictions[0]
np.argmax(predictions[0]) # pick the highest chance
test_labels[0]

# model.predict(array([hist[0]]))

# if not os.path.exists(run_dir):
#     os.makedirs(run_dir)
#
# # Setup some printing magic
# # https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
# # https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python?noredirect=1&lq=1
# stdout_add_file(os.path.join(run_dir, 'log.txt'))
# # we want to see everything in the prints
# np.set_printoptions(linewidth=200, threshold=(history_length + 1) * history_length * feature_num) # unset with np.set_printoptions()
#
# # =========== data
# answer_snapshots = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_train_l{history_length}.txt'))
#
# # input and outputs
# seq_in = answer_snapshots
# # we're using an auto encoder so the input is the output
# seq_out = seq_in
#
# # define model
# model = lstm_autoencoder(lstm_layer_size, history_length, feature_num)
# optimizer = Adam(learning_rate=0.001, epsilon=1e-7) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
# model.compile(optimizer=optimizer, loss='mse')
#
# # fit model
# model.fit(seq_in, seq_out, epochs=epochs, verbose=2)
#
#
# # print(answer_snapshots[:21])
# pred_seq_in = answer_snapshots[15:16]
# print("pred_seq_in")
# print(pred_seq_in)
# yhat = model.predict(pred_seq_in, verbose=0)
# # print(yhat)
#
# # # https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-above-a-specific-threshold
# # threshold_indices = (yhat < 0.03) & (yhat > -0.03)
# # yhat_zeroed = np.copy(yhat)
# # yhat_zeroed[threshold_indices] = 0
# # print(yhat_zeroed)
#
# yhat_round = np.around(yhat, decimals=1)
# print("yhat_round")
# print(yhat_round)
#
# compare = np.around(np.subtract(pred_seq_in, yhat_round), decimals=1)
# print("compare")
# print(compare)
#
# end = datetime.datetime.now()
# difference = end - start
#
# print(f'start      {start}')
# print(f'end        {end}')
# print(f'difference {difference}')
#
# stdout_reset()
#
# # https://www.tensorflow.org/guide/keras/save_and_serialize
# model.save(os.path.join(run_dir, 'model.h5'))

