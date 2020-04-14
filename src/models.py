import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

def lstm_autoencoder(encoding_size, seq_length, feature_num, return_sequences=True):
    # https://machinelearningmastery.com/lstm-autoencoders/
    # model = Sequential()
    # model.add(LSTM(encoding_size, activation='relu', input_shape=(seq_length, feature_num)))
    # model.add(RepeatVector(seq_length))
    # model.add(LSTM(encoding_size, activation='relu', return_sequences=return_sequences))
    # model.add(TimeDistributed(Dense(feature_num)))

    # https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    outer_layer_size = encoding_size * 2
    inner_layer_size = encoding_size
    model = Sequential()
    model.add(LSTM(outer_layer_size, activation='relu', input_shape=(seq_length, feature_num), return_sequences=True))
    model.add(LSTM(inner_layer_size, activation='relu', return_sequences=False))
    model.add(RepeatVector(seq_length))
    model.add(LSTM(inner_layer_size, activation='relu', return_sequences=True))
    model.add(LSTM(outer_layer_size, activation='relu', return_sequences=return_sequences))
    model.add(TimeDistributed(Dense(feature_num)))
    return model

def lstm_autoencoder_embedding_layer(embedding_model):
    # return embedding_model.layers[0].output
    return embedding_model.layers[1].output