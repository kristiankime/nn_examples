# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import math
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
import joblib

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
run_dir = os.path.join('dashboards', f'pfa_dashboard', f'run_random_forest')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log.txt'))



# =======================
# Get the data
pfa_dashboard_diff_none_train = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_train.csv'), delimiter=",", header=None)
pfa_dashboard_diff_none_train_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_train_answers.csv'), delimiter=",", header=None)

# =======================
# Train random Forest
# https://stackoverflow.com/questions/40366443/random-forest-in-python-final-probabilities-in-classification-problems#40366707
# clf = RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1)
clf = RandomForestClassifier(random_state=23)
clf.fit(pfa_dashboard_diff_none_train, pfa_dashboard_diff_none_train_answers)
joblib.dump(clf, os.path.join(run_dir, 'random_forest.dump'), compress=3)

# =======================
# Create and store Training Set data
predictions_train = clf.predict(pfa_dashboard_diff_none_train)
predicted_probs_train = clf.predict_proba(pfa_dashboard_diff_none_train)
predicted_probs_0_train = [item[0] for item in predicted_probs_train]
predicted_probs_1_train = [item[1] for item in predicted_probs_train]

pfa_dashboard_diff_none_train['actual'] = pfa_dashboard_diff_none_train_answers
pfa_dashboard_diff_none_train['pred'] = predictions_train
pfa_dashboard_diff_none_train['probs_0'] = predicted_probs_0_train
pfa_dashboard_diff_none_train['probs_1'] = predicted_probs_1_train
pfa_dashboard_diff_none_train['accuracy'] = (pfa_dashboard_diff_none_train['actual'] == pfa_dashboard_diff_none_train['pred']).astype(int)
pfa_dashboard_diff_none_train.to_csv(os.path.join(run_dir, 'random_forest_train_outcome.csv'), index=False)
accuracy_train = (pfa_dashboard_diff_none_train['actual'] == pfa_dashboard_diff_none_train['pred'])
accuracy_train.sum() / len(accuracy_train)


# =======================
# Create and store Test Set data
pfa_dashboard_diff_none_test = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test.csv'), delimiter=",", header=None)
pfa_dashboard_diff_none_test_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test_answers.csv'), delimiter=",", header=None)

predictions_test = clf.predict(pfa_dashboard_diff_none_test)
predicted_probs_test = clf.predict_proba(pfa_dashboard_diff_none_test)
predicted_probs_0_test = [item[0] for item in predicted_probs_test]
predicted_probs_1_test = [item[1] for item in predicted_probs_test]

pfa_dashboard_diff_none_test['actual'] = pfa_dashboard_diff_none_test_answers
pfa_dashboard_diff_none_test['pred'] = predictions_test
pfa_dashboard_diff_none_test['probs_0'] = predicted_probs_0_test
pfa_dashboard_diff_none_test['probs_1'] = predicted_probs_1_test
pfa_dashboard_diff_none_test['accuracy'] = (pfa_dashboard_diff_none_test['actual'] == pfa_dashboard_diff_none_test['pred']).astype(int)
pfa_dashboard_diff_none_test.to_csv(os.path.join(run_dir, 'random_forest_test_outcome.csv'), index=False)
accuracy_test = (pfa_dashboard_diff_none_test['actual'] == pfa_dashboard_diff_none_test['pred'])
accuracy_test.sum() / len(accuracy_test)


# print(predicted_probs)
# # test = pd.concat([test, pd.DataFrame(predicted_probs, columns=['Col_0', 'Col_1'])], axis=1)
#
#
# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')
#
# stdout_reset()
# # https://www.tensorflow.org/guide/keras/save_and_serialize
# # model.save(os.path.join(run_dir, 'model.h5'))