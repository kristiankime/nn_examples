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

from logs import stdout_add_file
from logs import stdout_reset
from util import create_snapshots

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)

# =========== Overview
# parameterization
user_size = 100
history_length = 20 # 243 possible
feature_num = 29 # <correct or not> + <28 features>

lstm_layer_size = 300
epochs = 200

# output location
run_dir = os.path.join('runs', f'run_u{user_size}e{epochs}_t{history_length}f{feature_num}_l{lstm_layer_size}')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
# https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python?noredirect=1&lq=1
stdout_add_file(os.path.join(run_dir, 'log.txt'))

# =========== data
answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs' , 'answers_history.csv'))
answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])

users = answer_history_base['anon_id'].unique()
users_n = users[:user_size]
answer_history_n = answer_history_trim[answer_history_trim.anon_id.isin(users_n)]

answer_snapshots = create_snapshots(answer_history_n, length=history_length)

print("answer_snapshots")
print(answer_snapshots)

# input and outputs
seq_in = answer_snapshots
# we're using an auto encoder so the input is the output
seq_out = seq_in


# https://github.com/keras-team/keras/issues/4563


# define model
model = Sequential()
model.add(LSTM(lstm_layer_size, activation='relu', input_shape=(history_length, feature_num))) # https://stackoverflow.com/questions/58086601/xavier-initialization-in-tensorflow-2-0
model.add(RepeatVector(history_length))
model.add(LSTM(lstm_layer_size, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(feature_num)))

optimizer = Adam(learning_rate=0.001, epsilon=1e-7) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model.compile(optimizer=optimizer, loss='mse')

# fit model
model.fit(seq_in, seq_out, epochs=epochs, verbose=2)


# we want to see everything in the prints
np.set_printoptions(linewidth=200, threshold=21*20*29)
# np.set_printoptions()
# print(answer_snapshots[:21])

pred_seq_in = answer_snapshots[:1]
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

