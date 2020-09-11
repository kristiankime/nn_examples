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
feature_num = 27 # <correct or not> + <26 features>
lstm_layer_size = 80
lstm_epochs = 245

pred_model_layer_1 = 1024
pred_model_layer_2 = 256
pred_epochs = 80

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# model load dir
run_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')
# output location
run_dir = os.path.join('runs', f'run_results_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)
#
# stdout_add_file(os.path.join(run_dir, 'log.txt'))
#
snapshots_validate_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
snapshots_validate_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")

# =========== Load the prediction model ===========

# load model
# Recreate the exact same model purely from the file
model = keras.models.load_model(os.path.join(run_dir_load, f'model.h5'))

# =========== Probability version of the model ===========
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# Get predictions from the model
predictions = probability_model.predict(snapshots_validate_embedded)
correct_prediction = predictions[:,1].transpose()

# get the actual outcome
actual_outcome = snapshots_validate_labels.values[:,0].transpose()

# concatenate them together and save to disk
pred_vs_actual = np.column_stack((correct_prediction, actual_outcome))
pred_vs_actual_df = pd.DataFrame(data=pred_vs_actual, columns=['prob', 'correct'])
pred_vs_actual_df.to_csv(os.path.join(run_dir, f'pred_vs_actual.csv'), index=False)

# # np.argmax(predictions[0]) # pick the highest chance
# # test_labels[0]
#
#
# # =========== End Reporting ===========
# end = datetime.datetime.now()
# difference = end - start
#
# print(f'start      {start}')
# print(f'end        {end}')
# print(f'difference {difference}')
#
# stdout_reset()
#
