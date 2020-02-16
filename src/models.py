from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed

def lstm_autoencoder(lstm_layer_size, seq_length, feature_num, return_sequences=True):
    # https://machinelearningmastery.com/lstm-autoencoders/
    model = Sequential()
    model.add(LSTM(lstm_layer_size, activation='relu', input_shape=(seq_length, feature_num)))
    model.add(RepeatVector(seq_length))
    model.add(LSTM(lstm_layer_size, activation='relu', return_sequences=return_sequences))
    model.add(TimeDistributed(Dense(feature_num)))
    return model
