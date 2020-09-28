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
from util.nn import nn_dashboard, nn_dashboard_data

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
full_history_length = 243
model_history_length = 13 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
diff_num = 5
skill_num = 21
diff_and_skill_num = diff_num + skill_num
feature_num = 1 + diff_and_skill_num # <correct or not> + <26 features>
lstm_layer_size = 80
lstm_epochs = 245

pred_model_layer_1 = 1024
pred_model_layer_2 = 256
pred_epochs = 80

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# model load dir
nn_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')
# nn_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')

# output location
# run_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')
run_dir = os.path.join('dashboards', f'nn_dashboard', 'pick1')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log.txt'))



# =========== Load the prediction model ===========
# load model
# Recreate the exact same model purely from the file
model = keras.models.load_model(os.path.join(nn_dir_load, f'model.h5'))
# Probability version of the model
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# # ================= TRAIN ===============
# print(f"loading train data")
# snapshots_train_embedded = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_train_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip")
# snapshots_train_labels = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_train_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip")
# # ========= Dashboard and current question
# print(f"computing dashboard")
# snapshots_train_embedded_dashboard = [
#     nn_dashboard(ac, probability_model, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1)
#     for ac in snapshots_train_embedded.to_numpy()]
# print(f"saving dashboard")
# np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_train_s1.csv'), snapshots_train_embedded_dashboard, fmt='%1.4f', delimiter=",")
# # ========= answer / label for current question
# print(f"saving answers")
# np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_train_answers_s1.csv'), snapshots_train_labels, fmt='%1.4f', delimiter=",")


# ================= VALIDATE ===============
print(f"loading validate data")
snapshots_validate_embedded = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip", header=None)
snapshots_validate_labels = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip", header=None)
# ========= Dashboard and current question
print(f"computing dashboard")
snapshots_validate_embedded_dashboard = [
    nn_dashboard(ac, probability_model, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1)
    for ac in snapshots_validate_embedded.to_numpy()]
print(f"saving dashboard")
np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_validate_s1.csv'), snapshots_validate_embedded_dashboard, fmt='%1.4f', delimiter=",")
# ========= answer / label for current question
print(f"saving answers")
np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_validate_answers_s1.csv'), snapshots_validate_labels, fmt='%1.4f', delimiter=",")


# ================= TEST ===============
print(f"loading test data")
snapshots_test_embedded = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_test_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip", header=None)
snapshots_test_labels = pd.io.parsers.read_csv(os.path.join('outputs', 'pick1', f'snapshots_test_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}_s1.csv.gz'), delimiter=",", compression="gzip", header=None)
# ========= Dashboard and current question
print(f"computing dashboard")
snapshots_test_embedded_dashboard = [
    nn_dashboard(ac, probability_model, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1)
    for ac in snapshots_test_embedded.to_numpy()]
print(f"saving dashboard")
np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_test_s1.csv'), snapshots_test_embedded_dashboard, fmt='%1.4f', delimiter=",", header=None)
# ========= answer / label for current question
print(f"saving answers")
np.savetxt(os.path.join(run_dir, f'nn_dashboard_diff_none_test_answers_s1.csv'), snapshots_test_labels, fmt='%1.4f', delimiter=",", header=None)




# coef = pfa_coef_counts(pfa_coef())

# # ================= TRAIN ===============
# print(f"loading train data")
# answer_counts_train = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_train_l{full_history_length}.txt'))
# # ========= Dashboard and current question
# print(f"computing dashboard")
# answer_counts_train_dashboard = [ np.concatenate(
#     (pfa_dashboard(ac, coef, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1), np.array(ac[2][1:]))
# ) for ac in answer_counts_train]
# print(f"saving dashboard")
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_train.csv'), answer_counts_train_dashboard, fmt='%1.4f', delimiter=",")
# # ========= answer / label for current question
# print(f"saving answers")
# answer_counts_train_dashboard_answer = [np.array([ac[2][0]]) for ac in answer_counts_train]
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_train_answers.csv'), answer_counts_train_dashboard_answer, fmt='%1.4f', delimiter=",")
#
#
# # ================= VALIDATE ===============
# print(f"loading validate data")
# answer_counts_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_validate_l{full_history_length}.txt'))
# # ========= Dashboard and current question
# print(f"computing dashboard")
# answer_counts_validate_dashboard = [ np.concatenate(
#     (pfa_dashboard(ac, coef, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1), np.array(ac[2][1:]))
# ) for ac in answer_counts_validate]
# print(f"saving dashboard")
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_validate.csv'), answer_counts_validate_dashboard, fmt='%1.4f', delimiter=",")
# # ========= answer / label for current question
# print(f"saving answers")
# answer_counts_validate_dashboard_answer = [np.array([ac[2][0]]) for ac in answer_counts_validate]
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_validate_answers.csv'), answer_counts_validate_dashboard_answer, fmt='%1.4f', delimiter=",")
#
#
# # ================= TEST ===============
# print(f"loading test data")
# answer_counts_test = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_test_l{full_history_length}.txt'))
# # ========= Dashboard and current question
# print(f"computing dashboard")
# answer_counts_test_dashboard = [ np.concatenate(
#         (pfa_dashboard(ac, coef, num_diffs=0, num_skills=diff_and_skill_num, diff_ind=-1), np.array(ac[2][1:]))
#     ) for ac in answer_counts_test]
# print(f"saving dashboard")
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_test.csv'), answer_counts_test_dashboard, fmt='%1.4f', delimiter=",")
# # ========= answer / label for current question
# print(f"saving answers")
# answer_counts_test_dashboard_answer = [np.array([ac[2][0]]) for ac in answer_counts_test]
# np.savetxt(os.path.join(run_dir, f'pfa_dashboard_diff_none_test_answers.csv'), answer_counts_test_dashboard_answer, fmt='%1.4f', delimiter=",")




# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()

