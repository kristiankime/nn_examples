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
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
full_history_length = 243
model_history_length = 13 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
feature_num = 29 # <correct or not> + <28 features>
lstm_layer_size = 100
epochs = 240

# pred_model_layer_1 = 512
# pred_model_layer_2 = 128

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# output location
# run_dir = os.path.join('runs', f'run_embedded_l1{pred_model_layer_1}_l2{pred_model_layer_2}_e{epochs}')

# if not os.path.exists(run_dir):
#     os.makedirs(run_dir)
#
# stdout_add_file(os.path.join(run_dir, 'log.txt'))

# =========== Created Embedded History / labels ===========
# Get the model and switch to using the LSTM Layer as output
model_dir = os.path.join('runs', f'run_t{model_history_length}_l{lstm_layer_size}_e{epochs}')
embedding_model = keras.models.load_model(os.path.join(model_dir, f'model.h5'))
# connect the encoder LSTM as the output layer
embedding_model = Model(inputs=embedding_model.inputs, outputs=embedding_model.layers[0].output)

# ------- Training
# snapshots_train = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_train_l{full_history_length}_10p.txt'))
snapshots_train = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_train_l{full_history_length}.txt'))
(snapshots_train_embedded, snapshots_train_labels) = split_snapshot_history(embedding_model, snapshots_train, model_history_length)
# np.savetxt(os.path.join('outputs', f'snapshots_train_embedded_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv'), snapshots_train_embedded, fmt='%1.4f', delimiter=",")
# np.savetxt(os.path.join('outputs', f'snapshots_train_labels_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv'), snapshots_train_labels, fmt='%1.4f', delimiter=",")
np.savetxt(os.path.join('outputs', f'snapshots_train_embedded_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_train_embedded, fmt='%1.4f', delimiter=",")
np.savetxt(os.path.join('outputs', f'snapshots_train_labels_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_train_labels, fmt='%1.4f', delimiter=",")


# ------- Validation
snapshots_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_validate_l{full_history_length}.txt'))
(snapshots_validate_embedded, snapshots_validate_labels) = split_snapshot_history(embedding_model, snapshots_validate, model_history_length)
np.savetxt(os.path.join('outputs', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_validate_embedded, fmt='%1.4f', delimiter=",")
np.savetxt(os.path.join('outputs', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_validate_labels, fmt='%1.4f', delimiter=",")

# ------- Test
snapshots_test = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_test_l{full_history_length}.txt'))
(snapshots_test_embedded, snapshots_test_labels) = split_snapshot_history(embedding_model, snapshots_test, model_history_length)
np.savetxt(os.path.join('outputs', f'snapshots_test_embedded_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_test_embedded, fmt='%1.4f', delimiter=",")
np.savetxt(os.path.join('outputs', f'snapshots_test_labels_t{model_history_length}_l{lstm_layer_size}_e{epochs}.csv.gz'), snapshots_test_labels, fmt='%1.4f', delimiter=",")
















# np.savetxt(os.path.join(run_dir, f'snapshots_train_embedded.1_10.csv'), snapshots_train_embedded[1:10], fmt='%1.4f', delimiter=",")
# np.savetxt(os.path.join(run_dir, f'snapshots_train_embedded.12_16.csv'), snapshots_train_embedded[12:16], fmt='%1.4f', delimiter=",")

# # snapshot = snapshots_train[100:101][0]
#
# # =========== Build the prediction model ===========
# # https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# # https://www.tensorflow.org/tutorials/keras/classification?hl=nb
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(1928,)),  # input shape required
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(2) # Binary output
# ])
# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#
# # train the model
# train_inputs = snapshots_train_embedded
# train_labels = snapshots_train_labels
#
# model_fit_history = model.fit(train_inputs, train_labels, epochs=10)
# # model.fit(train_inputs, train_labels, epochs=10, verbose=2)
#
# # =========== Validate ===========
# snapshots_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_validate_l{full_history_length}_10p.txt'))
# (snapshots_validate_embedded, snapshots_validate_labels) = split_snapshot_history(embedding_model, snapshots_validate, model_history_length)
#
# test_inputs = snapshots_validate_embedded
# test_labels = snapshots_validate_labels
# test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)
#
#
# # =========== Probability version of the model ===========
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_inputs)
# predictions[0]
# np.argmax(predictions[0]) # pick the highest chance
# test_labels[0]