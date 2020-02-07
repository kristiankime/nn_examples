# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
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

# cwd = os.getcwd()
# print(f"cwd={cwd}")

start = datetime.datetime.now()
# the input values are random so set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)

# =========== Overview
# lstm autoencoder predict sequence
samples = 4000
# timesteps = 243 # 243 possible
timesteps = 40
# feature_num = 29 # <correct or not> + <28 features>
feature_num = 29 # <correct or not> + <28 features>

lstm_layer_size = 300
epochs = 200

run_dir = os.path.join('runs', f'run_s{samples}e{epochs}_t{timesteps}f{feature_num}_l{lstm_layer_size}')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
# https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python?noredirect=1&lq=1
stdout_add_file(os.path.join(run_dir, 'log.txt'))

# =========== sample data generation functions
def history(max_timesteps, feature_num):
    history_length = np.random.random_integers(low=1, high=max_timesteps)
    filled_shape = (history_length, feature_num)
    padded_shape = (max_timesteps, feature_num)
    filled_history = np.random.random_integers(low=0, high=1, size=filled_shape).astype(np.float32)

    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    padded_history = np.zeros(padded_shape).astype(np.float32)
    padded_history[:filled_shape[0],:filled_shape[1]] = filled_history
    return padded_history


def input_data(samples, timesteps, feature_num):
    return np.array([history(timesteps, feature_num) for i in range(samples)])


seq_in = input_data(samples, timesteps, feature_num)

# https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss#51005846
# print("nan count " + str(np.count_nonzero(np.isnan(seq_in))))

# we're using an auto encoder so the input is the output
seq_out = seq_in


# https://github.com/keras-team/keras/issues/4563


# define model
model = Sequential()
model.add(LSTM(lstm_layer_size, activation='relu', input_shape=(timesteps,feature_num))) # https://stackoverflow.com/questions/58086601/xavier-initialization-in-tensorflow-2-0
model.add(RepeatVector(timesteps))
model.add(LSTM(lstm_layer_size, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(feature_num)))

optimizer = Adam(learning_rate=0.001, epsilon=1e-7) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model.compile(optimizer=optimizer, loss='mse')
# model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(seq_in, seq_out, epochs=epochs, verbose=2)

pred_seq_in = array([history(timesteps, feature_num)])
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

