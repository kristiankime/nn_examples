# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import math
import decimal
import matplotlib.pyplot as plt
import scipy.stats as st

from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from util.logs import stdout_add_file, stdout_reset


def drange_inc(x, y, jump):
    while x <= y:
        yield float(x)
        x += decimal.Decimal(jump)


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
pfa_dashboard_diff_none_train: pd.DataFrame = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_train.csv'), delimiter=",", header=None)
pfa_dashboard_diff_none_train_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_train_answers.csv'), delimiter=",", header=None)

# =======================
# Train random Forest
# https://stackoverflow.com/questions/40366443/random-forest-in-python-final-probabilities-in-classification-problems#40366707
# clf = RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1)
clf = RandomForestClassifier(random_state=23)
# clf.fit(pfa_dashboard_diff_none_train, pfa_dashboard_diff_none_train_answers)
# joblib.dump(clf, os.path.join(run_dir, 'random_forest.dump'), compress=3)


# https://stackoverflow.com/questions/38151615/specific-cross-validation-with-random-forest#38155664

param_grid = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [2, 5, 7, 9]
}

grid_clf = GridSearchCV(clf, param_grid, cv=10)
grid_clf.fit(pfa_dashboard_diff_none_train, pfa_dashboard_diff_none_train_answers)

# # You can then get the best model using
# grid_clf.best_estimator_  # RandomForestClassifier(max_depth=9, n_estimators=20, random_state=23)
# # and the best parameters using
# grid_clf.best_params_  # {'max_depth': 9, 'n_estimators': 20}
# # Similarly you can get the grid scores using
# grid_clf.cv_results_

joblib.dump(grid_clf.best_estimator_, os.path.join(run_dir, 'random_forest_best.dump'), compress=3)



# ============ move to another file?
classifier: RandomForestClassifier = joblib.load(os.path.join(run_dir, 'random_forest_best.dump'))

# 10-Fold Cross validation
# print(np.mean(cross_val_score(clf, pfa_dashboard_diff_none_train, pfa_dashboard_diff_none_train_answers, cv=10)))

bins = list(drange_inc(0, 1, '0.05')) # 5% point bin size
bin_labels = list(range(1, 21))

# =======================
# Create and store Training Vs Actual Comparison data
predictions_train = classifier.predict(pfa_dashboard_diff_none_train)
predicted_probs_train = classifier.predict_proba(pfa_dashboard_diff_none_train)
predicted_probs_0_train = [item[0] for item in predicted_probs_train]
predicted_probs_1_train = [item[1] for item in predicted_probs_train]

pfa_dashboard_diff_none_train_stats: pd.DataFrame = pfa_dashboard_diff_none_train.copy()
pfa_dashboard_diff_none_train_stats['actual'] = pfa_dashboard_diff_none_train_answers
pfa_dashboard_diff_none_train_stats['pred'] = predictions_train
pfa_dashboard_diff_none_train_stats['probs_0'] = predicted_probs_0_train
pfa_dashboard_diff_none_train_stats['probs_1'] = predicted_probs_1_train
pfa_dashboard_diff_none_train_stats['accuracy'] = (pfa_dashboard_diff_none_train_stats['actual'] == pfa_dashboard_diff_none_train_stats['pred']).astype(int)
pfa_dashboard_diff_none_train_stats['binned_ind'] = pd.cut(pfa_dashboard_diff_none_train_stats['probs_1'], bins=bins, labels=bin_labels)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
pfa_dashboard_diff_none_train_stats['binned_range'] = pd.cut(pfa_dashboard_diff_none_train_stats['probs_1'], bins=bins)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
pfa_dashboard_diff_none_train_stats.to_csv(os.path.join(run_dir, 'random_forest_train_outcome.csv'), index=False)


def binning(g):
    return pd.Series(data={'actual': g.actual.sum(), 'count': len(g.index)})

pfa_dashboard_diff_none_train_gb = pfa_dashboard_diff_none_train_stats.groupby(by=['binned_range']).apply(binning).reset_index()
pfa_dashboard_diff_none_train_gb['rate'] = pfa_dashboard_diff_none_train_gb['binned_range'].apply(lambda x: x.right)
pfa_dashboard_diff_none_train_gb['expected'] = pfa_dashboard_diff_none_train_gb['count'] * pfa_dashboard_diff_none_train_gb['rate']
pfa_dashboard_diff_none_train_gb['actual_rate'] = pfa_dashboard_diff_none_train_gb['actual'] / pfa_dashboard_diff_none_train_gb['count']
# R = pfa_dashboard_diff_none_train_gb['rate']
# O = pfa_dashboard_diff_none_train_gb['actual']
# E = pfa_dashboard_diff_none_train_gb['expected']
# OE = O - E
# C2 = (OE * OE) / E
# pfa_dashboard_diff_none_train_gb.replace(np.nan, 0, inplace=True)
pfa_dashboard_diff_none_train_gb.to_csv(os.path.join(run_dir, 'random_forest_train_outcome_gb.csv'), index=False)


plt.plot(pfa_dashboard_diff_none_train_gb['rate'], pfa_dashboard_diff_none_train_gb['actual_rate'], '-o')
plt.plot(pfa_dashboard_diff_none_train_gb['rate'], pfa_dashboard_diff_none_train_gb['rate'], '-o')
plt.savefig(os.path.join(run_dir, 'random_forest_train_outcome_gb.pdf'), bbox_inches='tight')
#
# pfa_dashboard_diff_none_train_gb_nn = pfa_dashboard_diff_none_train_gb.dropna()
#
# # # https://stackoverflow.com/questions/51894150/python-chi-square-goodness-of-fit-test-to-get-the-best-distribution
# # # c, p = st.chisquare(observed_values, expected_values, ddof=len(param))
# # # https://www.gigacalculator.com/calculators/chi-square-to-p-value-calculator.php
# # c, p = st.chisquare(pfa_dashboard_diff_none_train_gb_nn['actual_rate'], pfa_dashboard_diff_none_train_gb_nn['expected'], ddof=len(pfa_dashboard_diff_none_train_gb_nn.index)-1)



#
#
#
#
#
# # =======================
# # Create and store Validation Set data
# pfa_dashboard_diff_none_test = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test.csv'), delimiter=",", header=None)
# pfa_dashboard_diff_none_test_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'pfa_dashboard', f'pfa_dashboard_diff_none_test_answers.csv'), delimiter=",", header=None)
#
# predictions_test = classifier.predict(pfa_dashboard_diff_none_test)
# predicted_probs_test = classifier.predict_proba(pfa_dashboard_diff_none_test)
# predicted_probs_0_test = [item[0] for item in predicted_probs_test]
# predicted_probs_1_test = [item[1] for item in predicted_probs_test]
#
# pfa_dashboard_diff_none_test_stats = pfa_dashboard_diff_none_test.copy()
# pfa_dashboard_diff_none_test_stats['actual'] = pfa_dashboard_diff_none_test_answers
# pfa_dashboard_diff_none_test_stats['pred'] = predictions_test
# pfa_dashboard_diff_none_test_stats['probs_0'] = predicted_probs_0_test
# pfa_dashboard_diff_none_test_stats['probs_1'] = predicted_probs_1_test
# pfa_dashboard_diff_none_test_stats['accuracy'] = (pfa_dashboard_diff_none_test['actual'] == pfa_dashboard_diff_none_test['pred']).astype(int)
# pfa_dashboard_diff_none_test_stats['binned_range'] = pd.cut(pfa_dashboard_diff_none_test_stats['probs_1'], bins=bins, labels=bin_labels)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
# pfa_dashboard_diff_none_test_stats['binned_inx'] = pd.cut(pfa_dashboard_diff_none_test_stats['probs_1'], bins=bins)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
#
# pfa_dashboard_diff_none_test.to_csv(os.path.join(run_dir, 'random_forest_test_outcome.csv'), index=False)
# accuracy_test = (pfa_dashboard_diff_none_test['actual'] == pfa_dashboard_diff_none_test['pred'])
# accuracy_test.sum() / len(accuracy_test)


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