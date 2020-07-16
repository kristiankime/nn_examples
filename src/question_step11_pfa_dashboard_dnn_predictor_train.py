# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import math
from tensorflow import keras

from util.logs import stdout_add_file, stdout_reset

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
full_history_length = 243
model_history_length = 13 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
# feature_num = 27 # <correct or not> + <26 features>
# lstm_layer_size = 80
# lstm_epochs = 245

input_size = 26 + 26 # 26 dashboard + 26 current question skills

pred_model_layer_1 = 256
pred_model_layer_2 = 256
pred_epochs = 100

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * input_size) # unset with np.set_printoptions()

# output location
run_dir = os.path.join('dashboards', f'pfa_dashboard', f'run_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log.txt'))


# Get the data
pfa_dashboard_diff_none_test = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test.csv'), delimiter=",", header=None)
pfa_dashboard_diff_none_test_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test_answers.csv'), delimiter=",", header=None)

# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_test.csv'), answer_counts_test_dashboard, fmt='%1.4f', delimiter=",")
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_test_answers.csv'), answer_counts_test_dashboard_answer, fmt='%1.4f', delimiter=",")
# snapshots_train_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_train_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
# snapshots_train_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_train_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")

# =========== Build the prediction model ===========
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/keras/classification?hl=nb

model = tf.keras.Sequential([
    tf.keras.layers.Dense(pred_model_layer_1, activation=tf.nn.relu, input_shape=(input_size,)),  # input shape required
    tf.keras.layers.Dense(pred_model_layer_2, activation=tf.nn.relu),
    tf.keras.layers.Dense(2) # Binary output
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# train the model
train_inputs = pfa_dashboard_diff_none_test
train_labels = pfa_dashboard_diff_none_test_answers

model_fit_history = model.fit(train_inputs, train_labels, epochs=pred_epochs, verbose=2)
# model.fit(train_inputs, train_labels, epochs=pred_epochs, verbose=2)


# =========== Validate ===========
# snapshots_validate_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
# snapshots_validate_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
#
# test_inputs = snapshots_validate_embedded
# test_labels = snapshots_validate_labels
# test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)


# # =========== Probability version of the model ===========
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_inputs)
# predictions[0]
# np.argmax(predictions[0]) # pick the highest chance
# test_labels[0]

# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()
# https://www.tensorflow.org/guide/keras/save_and_serialize
model.save(os.path.join(run_dir, 'model.h5'))