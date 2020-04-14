# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from tensorflow import keras

from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

from util.logs import stdout_add_file, stdout_reset
from util.util import group_snapshots, read_numpy_3d_array_from_txt

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
# user_size = 800 # 3285
history_length = 25 # 243 possible
feature_num = 27 # <correct or not> + <26 features>

lstm_layer_size = 64
epochs = 240

# output location
run_dir = os.path.join('runs', f'run_t{history_length}_l{lstm_layer_size}_e{epochs}')
score_dir = os.path.join('runs', f'run_t{history_length}_l{lstm_layer_size}_e{epochs}_score')

if not os.path.exists(score_dir):
    os.makedirs(score_dir)

# Setup some printing magic
# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
# https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python?noredirect=1&lq=1
stdout_add_file(os.path.join(score_dir, 'score.txt'))
# we want to see everything in the prints
np.set_printoptions(linewidth=200, threshold=(history_length + 1) * history_length * feature_num) # unset with np.set_printoptions()

# =========== data
answer_snapshots = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_validate_l{history_length}.txt'))

# input and outputs
seq_in = answer_snapshots
# we're using an auto encoder so the input is the output
seq_out = seq_in


# https://github.com/keras-team/keras/issues/4563


# load model
# Recreate the exact same model purely from the file
model = keras.models.load_model(os.path.join(run_dir, f'model.h5'))


# eval model
print('# Evaluate on test data')
results = model.evaluate(seq_in, seq_out, batch_size=128, verbose=2)
print('test loss, test acc:')
print(results)


# print(answer_snapshots[:21])
pred_seq_in = answer_snapshots[15:16]
print("pred_seq_in")
print(pred_seq_in)
yhat = model.predict(pred_seq_in, verbose=0)

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
