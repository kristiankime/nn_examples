# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model

# the input values are random so set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)

# =========== Overview
# lstm autoencoder predict sequence

# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

#plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')

# fit model
model.fit(seq_in, seq_out, epochs=300, verbose=0)

# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat[0,:,0])