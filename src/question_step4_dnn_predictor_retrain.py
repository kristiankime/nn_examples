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

from util.logs import stdout_add_file, stdout_reset
from util.util import group_snapshots, read_numpy_3d_array_from_txt
from models import lstm_autoencoder
from util.data import split_snapshot_history_single, split_snapshot_history

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
lstm_epochs = 240

pred_model_layer_1 = 1024
pred_model_layer_2 = 256
pred_start_epochs = 40
pred_end_epochs = 80

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# output location
run_dir_old = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_start_epochs}')
run_dir_new = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_end_epochs}')

if not os.path.exists(run_dir_new):
    os.makedirs(run_dir_new)

stdout_add_file(os.path.join(run_dir_new, 'log.txt'))

snapshots_train_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_train_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
snapshots_train_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_train_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")

# =========== Load the prediction model ===========

# load model
# Recreate the exact same model purely from the file
model = keras.models.load_model(os.path.join(run_dir_old, f'model.h5'))

# train the model
train_inputs = snapshots_train_embedded
train_labels = snapshots_train_labels

model_fit_history = model.fit(train_inputs, train_labels, epochs=(pred_end_epochs - pred_start_epochs), verbose=2)
# model.fit(train_inputs, train_labels, epochs=pred_epochs, verbose=2)

# =========== Validate ===========
snapshots_validate_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
snapshots_validate_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")

test_inputs = snapshots_validate_embedded
test_labels = snapshots_validate_labels
test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()

# https://www.tensorflow.org/guide/keras/save_and_serialize
model.save(os.path.join(run_dir_new, 'model.h5'))

# # =========== Validate ===========
# snapshots_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_validate_l{full_history_length}_10p.txt'))
# (snapshots_validate_embedded, snapshots_validate_labels) = split_snapshot_history(embedding_model, snapshots_validate, model_history_length)

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