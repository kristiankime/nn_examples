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
import scipy.stats as stats

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

# pred_model_layer_1 = 256
# pred_model_layer_2 = 256
# pred_epochs = 100

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * input_size) # unset with np.set_printoptions()

# output location
run_dir = os.path.join('dashboards', f'nn_dashboard', f'run_random_forest')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log_confirm.txt'))

# ============
classifier: RandomForestClassifier = joblib.load(os.path.join(run_dir, 'random_forest_best.dump'))

# 10-Fold Cross validation
# print(np.mean(cross_val_score(clf, nn_dashboard_diff_none_train, nn_dashboard_diff_none_train_answers, cv=10)))

bins = list(drange_inc(0, 1, '0.05')) # 5% point bin size
bin_labels = list(range(1, 21))

# def binning(g):
#     return pd.Series(data={'actual': g.actual.sum(), 'count': len(g.index)})

# =======================
# Create and store Validation Set data
nn_dashboard_diff_none_validate = pd.io.parsers.read_csv(os.path.join('dashboards', f'nn_dashboard', 'pick1', f'nn_dashboard_diff_none_validate_s1.csv'), delimiter=",", header=None)
nn_dashboard_diff_none_validate_answers = pd.io.parsers.read_csv(os.path.join('dashboards', f'nn_dashboard', 'pick1', f'nn_dashboard_diff_none_validate_answers_s1.csv'), delimiter=",", header=None)

predictions_validate = classifier.predict(nn_dashboard_diff_none_validate)
predicted_probs_validate = classifier.predict_proba(nn_dashboard_diff_none_validate)
predicted_probs_0_validate = [item[0] for item in predicted_probs_validate]
predicted_probs_1_validate = [item[1] for item in predicted_probs_validate]

nn_dashboard_diff_none_validate_stats = nn_dashboard_diff_none_validate.copy()
nn_dashboard_diff_none_validate_stats['actual'] = nn_dashboard_diff_none_validate_answers
nn_dashboard_diff_none_validate_stats['pred'] = predictions_validate
nn_dashboard_diff_none_validate_stats['probs_0'] = predicted_probs_0_validate
nn_dashboard_diff_none_validate_stats['probs_1'] = predicted_probs_1_validate
nn_dashboard_diff_none_validate_stats['accuracy'] = (nn_dashboard_diff_none_validate_stats['actual'] == nn_dashboard_diff_none_validate_stats['pred']).astype(int)
nn_dashboard_diff_none_validate_stats['binned_ind'] = pd.cut(nn_dashboard_diff_none_validate_stats['probs_1'], bins=bins, labels=bin_labels)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
nn_dashboard_diff_none_validate_stats['binned_range'] = pd.cut(nn_dashboard_diff_none_validate_stats['probs_1'], bins=bins)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
nn_dashboard_diff_none_validate_stats.to_csv(os.path.join(run_dir, 'random_forest_validate_outcome.csv'), index=False)


def binning(g):
    return pd.Series(data={'actual': g.actual.sum(), 'count': len(g.index)})


# ======== with all
nn_dashboard_diff_none_validate_gb = nn_dashboard_diff_none_validate_stats.groupby(by=['binned_range']).apply(binning).reset_index()
nn_dashboard_diff_none_validate_gb['rate'] = nn_dashboard_diff_none_validate_gb['binned_range'].apply(lambda x: x.right)
nn_dashboard_diff_none_validate_gb['expected'] = nn_dashboard_diff_none_validate_gb['count'] * nn_dashboard_diff_none_validate_gb['rate']
nn_dashboard_diff_none_validate_gb['actual_rate'] = nn_dashboard_diff_none_validate_gb['actual'] / nn_dashboard_diff_none_validate_gb['count']
nn_dashboard_diff_none_validate_gb.to_csv(os.path.join(run_dir, 'random_forest_validate_outcome_gb.csv'), index=False)

# nn_dashboard_diff_none_validate_gb.dropna()
# R = nn_dashboard_diff_none_train_gb['rate']
O = nn_dashboard_diff_none_validate_gb['actual']
E = nn_dashboard_diff_none_validate_gb['expected']
OE = O - E
C2 = (OE * OE) / E

c2_stats = stats.chisquare(f_obs=nn_dashboard_diff_none_validate_gb.dropna()['actual'], f_exp=nn_dashboard_diff_none_validate_gb.dropna()['expected'])
print(c2_stats)
# stats.chisquare(f_obs=(nn_dashboard_diff_none_validate_gb.dropna()['actual']+2), f_exp=nn_dashboard_diff_none_validate_gb.dropna()['actual'])
# stats.chisquare(f_obs=[1, 2, 3, 5, 5, 6], f_exp=[1, 2, 3, 4, 5, 6])


plt.plot(nn_dashboard_diff_none_validate_gb['rate'], nn_dashboard_diff_none_validate_gb['actual_rate'], '-o')
plt.plot(nn_dashboard_diff_none_validate_gb['rate'], nn_dashboard_diff_none_validate_gb['rate'], '-o')
plt.savefig(os.path.join(run_dir, 'random_forest_validate_outcome_gb.pdf'), bbox_inches='tight')

# ======== with 100
nn_dashboard_diff_none_validate_stats_100 = nn_dashboard_diff_none_validate_stats.sample(100)

nn_dashboard_diff_none_validate_gb_100 = nn_dashboard_diff_none_validate_stats_100.groupby(by=['binned_range']).apply(binning).reset_index()
nn_dashboard_diff_none_validate_gb_100['rate'] = nn_dashboard_diff_none_validate_gb_100['binned_range'].apply(lambda x: x.right)
nn_dashboard_diff_none_validate_gb_100['expected'] = nn_dashboard_diff_none_validate_gb_100['count'] * nn_dashboard_diff_none_validate_gb_100['rate']
nn_dashboard_diff_none_validate_gb_100['actual_rate'] = nn_dashboard_diff_none_validate_gb_100['actual'] / nn_dashboard_diff_none_validate_gb_100['count']
nn_dashboard_diff_none_validate_gb_100.to_csv(os.path.join(run_dir, 'random_forest_validate_outcome_gb_100.csv'), index=False)

c2_stats_100 = stats.chisquare(f_obs=nn_dashboard_diff_none_validate_gb_100.dropna()['actual'], f_exp=nn_dashboard_diff_none_validate_gb_100.dropna()['expected'])
print(c2_stats_100)

plt.plot(nn_dashboard_diff_none_validate_gb_100['rate'], nn_dashboard_diff_none_validate_gb_100['actual_rate'], '-o')
plt.plot(nn_dashboard_diff_none_validate_gb_100['rate'], nn_dashboard_diff_none_validate_gb_100['rate'], '-o')
plt.savefig(os.path.join(run_dir, 'random_forest_validate_outcome_gb_100.pdf'), bbox_inches='tight')


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