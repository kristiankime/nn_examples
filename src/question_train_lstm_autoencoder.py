# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from logs import stdout_add_file, stdout_reset
from util import group_snapshots, read_numpy_3d_array_from_txt
from models import lstm_autoencoder

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
# user_size = 500 # 3285
history_length = 25 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
feature_num = 29 # <correct or not> + <28 features>

lstm_layer_size = 100
epochs = 20

# output location
run_dir = os.path.join('runs', f'run_t{history_length}_l{lstm_layer_size}_e{epochs}')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# Setup some printing magic
# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
# https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python?noredirect=1&lq=1
stdout_add_file(os.path.join(run_dir, 'log.txt'))
# we want to see everything in the prints
np.set_printoptions(linewidth=200, threshold=(history_length + 1) * history_length * feature_num) # unset with np.set_printoptions()

# =========== data
answer_snapshots = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_train_l{history_length}.txt'))

# input and outputs
seq_in = answer_snapshots
# we're using an auto encoder so the input is the output
seq_out = seq_in

# define model
model = lstm_autoencoder(lstm_layer_size, history_length, feature_num)
optimizer = Adam(learning_rate=0.001, epsilon=1e-7) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model.compile(optimizer=optimizer, loss='mse')

# fit model
model.fit(seq_in, seq_out, epochs=epochs, verbose=2)


# print(answer_snapshots[:21])
pred_seq_in = answer_snapshots[15:16]
print("pred_seq_in")
print(pred_seq_in)
yhat = model.predict(pred_seq_in, verbose=0)
# print(yhat)

# # https://stackoverflow.com/questions/28430904/set-numpy-array-elements-to-zero-if-they-are-above-a-specific-threshold
# threshold_indices = (yhat < 0.03) & (yhat > -0.03)
# yhat_zeroed = np.copy(yhat)
# yhat_zeroed[threshold_indices] = 0
# print(yhat_zeroed)

yhat_round = np.around(yhat, decimals=1)
print("yhat_round")
print(yhat_round)

compare = np.around(np.subtract(pred_seq_in, yhat_round), decimals=1)
print("compare")
print(compare)

end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()

# https://www.tensorflow.org/guide/keras/save_and_serialize
model.save(os.path.join(run_dir, 'model.h5'))

