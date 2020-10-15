# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats

from tensorflow import keras

from numpy import array
from tensorflow.keras.models import Model

from util.logs import stdout_add_file, stdout_reset
from util.util import pick_1_in_group
from util.util import group_snapshots, read_numpy_3d_array_from_txt, drange_inc
from models import lstm_autoencoder
from util.data import split_snapshot_history_single, split_snapshot_history
from util.pfa import pfa_prediction, pfa_coef_counts, pfa_coef
from util.util import drange_inc, add_binning_cols, binned_counts

start = datetime.datetime.now()
# set the seed for reproducibility
tf.random.set_seed(23)
np.random.seed(23)  # pandas uses numpy

# =========== Overview
# parameterization
full_history_length = 243
model_history_length = 13 # 243 possible but can't do all of them sometimes see this https://github.com/keras-team/keras/issues/4563 and sometimes the results are just bad
feature_num = 27 # <correct or not> + <26 features>
lstm_layer_size = 80
lstm_epochs = 245

pred_model_layer_1 = 1024
pred_model_layer_2 = 256
pred_epochs = 80

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# output location
# history_ids_test.to_csv(os.path.join('outputs', f'history_test_l{history_length}.csv'), index=False)
result_dir = os.path.join('results', f'pfa_vs_dnn')

if not os.path.exists(result_dir):
    os.makedirs(result_dir)



history_ids_validate = pd.read_csv(os.path.join('outputs', f'history_validate_l{full_history_length}.csv'))
keep_cols = ['actual', 'probs_1']
pfa_pred_validate = pd.read_csv(os.path.join(os.path.join('runs', f'run_results_pfa'), f'pfa_pred_vs_actual_validate.csv'))
pfa_dash_pred_validate = pd.read_csv(os.path.join('dashboards', f'pfa_dashboard', 'run_random_forest', 'pred_vs_actual', f'random_forest_validate_pva.csv'))[keep_cols].rename(columns={'probs_1': 'pfa_d_pred', 'actual': 'pfa_d_cor'})
dnn_pred_validate = pd.read_csv(os.path.join(os.path.join('runs', f'run_results_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}'), f'nn_pred_vs_actual_validate.csv')).rename(columns={'prob': 'dnn_pred', 'correct': 'dnn_cor'})
dnn_dash_pred_validate = pd.read_csv(os.path.join('dashboards', f'nn_dashboard', 'run_random_forest', 'pred_vs_actual', f'random_forest_validate_pva.csv'))[keep_cols].rename(columns={'probs_1': 'dnn_d_pred', 'actual': 'dnn_d_cor'})

pfa_vs_dnn = pd.concat([
    history_ids_validate,
    pfa_pred_validate,
    pfa_dash_pred_validate,
    dnn_pred_validate,
    dnn_dash_pred_validate
    ],
    axis=1)


pfa_vs_dnn.to_csv(os.path.join(result_dir, f'pfa_pred_vs_dnn_pred_w_dash_validate.csv'), index=False)

bins = list(drange_inc(0, 1, '0.05')) # 5% point bin size
bin_labels = list(range(1, 21))
base_cols = ['pfa', 'pfa_d', 'dnn', 'dnn_d']
correct_cols = [c + "_cor" for c in base_cols]
prediction_cols = [c + "_pred" for c in base_cols]

# correct_cols = ['pfa_cor', 'dnn_d_cor', 'dnn_cor', 'pfa_d_cor']
# prediction_cols = ['pfa_pred', 'dnn_d_pred', 'dnn_pred', 'pfa_d_pred']

pfa_vs_dnn_binned = pfa_vs_dnn.copy()
pfa_vs_dnn_binned = pfa_vs_dnn_binned.drop(columns=correct_cols)

# pfa_vs_dnn_binned.to_csv(os.path.join(result_dir, f'pfa_pred_vs_dnn_pred_w_dash_validate_no_cor.csv'), index=False)


for prob_col in prediction_cols:
    add_binning_cols(pfa_vs_dnn_binned, prob_col=prob_col, prefix=prob_col, bins=bins, bin_labels=bin_labels)

# prediction_cols = ['pfa_pred', 'dnn_d_pred', 'dnn_pred', 'pfa_d_pred']

pfa_vs_dnn_binned.to_csv(os.path.join(result_dir, f'pfa_pred_vs_dnn_pred_w_dash_bin_validate.csv'), index=False)


pfa_gb = binned_counts(pfa_vs_dnn_binned, actual_col='correct', bin_col='pfa_pred' + '_range')
pfa_d_gb = binned_counts(pfa_vs_dnn_binned, actual_col='correct', bin_col='pfa_d_pred' + '_range')
dnn_gb = binned_counts(pfa_vs_dnn_binned, actual_col='correct', bin_col='dnn_pred' + '_range')
dnn_d_gb = binned_counts(pfa_vs_dnn_binned, actual_col='correct', bin_col='dnn_d_pred' + '_range')

fig1, ax1 = plt.subplots()
ax1.plot(pfa_gb['rate'], pfa_gb['rate'], '-o', label='rate')
ax1.plot(pfa_gb['rate'], pfa_gb['actual_rate'], '-o', label='pfa')
ax1.plot(pfa_d_gb['rate'], pfa_d_gb['actual_rate'], '-o', label='pfa_d')
ax1.plot(dnn_gb['rate'], dnn_gb['actual_rate'], '-o', label='dnn')
ax1.plot(dnn_d_gb['rate'], dnn_d_gb['actual_rate'], '-o', label='dnn_d')
ax1.legend(loc='upper left')
fig1.savefig(os.path.join(result_dir, 'pfa_dnn_dash_compare_all.pdf'), bbox_inches='tight')


# Do Chi^2 on 1 user from each group
pfa_vs_dnn_binned_all_s1 = pick_1_in_group(pfa_vs_dnn_binned, 'anon_id', 'seq')
pfa_vs_dnn_binned_all_s1.to_csv(os.path.join(result_dir, f'pfa_pred_vs_dnn_pred_w_dash_bin_validate_all_s1.csv'), index=False)
picked = list(pfa_vs_dnn_binned_all_s1['seq'])
# use the list to select snapshots
pfa_vs_dnn_binned_s1 = pfa_vs_dnn_binned_all_s1[picked].sample(n=110)

pfa_vs_dnn_binned_s1.to_csv(os.path.join(result_dir, f'pfa_pred_vs_dnn_pred_w_dash_bin_validate_s1.csv'), index=False)
# gb for the 100
pfa_gb_s1 = binned_counts(pfa_vs_dnn_binned_s1, actual_col='correct', bin_col='pfa_pred' + '_range')
pfa_d_gb_s1 = binned_counts(pfa_vs_dnn_binned_s1, actual_col='correct', bin_col='pfa_d_pred' + '_range')
dnn_gb_s1 = binned_counts(pfa_vs_dnn_binned_s1, actual_col='correct', bin_col='dnn_pred' + '_range')
dnn_d_gb_s1 = binned_counts(pfa_vs_dnn_binned_s1, actual_col='correct', bin_col='dnn_d_pred' + '_range')

c2_stats_pfa = stats.chisquare(f_obs=pfa_gb_s1.dropna()['actual'], f_exp=pfa_gb_s1.dropna()['count_expected'])
print(f'c2_stats_pfa {c2_stats_pfa}')
c2_stats_pfa_d = stats.chisquare(f_obs=pfa_d_gb_s1.dropna()['actual'], f_exp=pfa_d_gb_s1.dropna()['count_expected'])
print(f'c2_stats_pfa_d {c2_stats_pfa_d}')
c2_stats_dnn = stats.chisquare(f_obs=dnn_gb_s1.dropna()['actual'], f_exp=dnn_gb_s1.dropna()['count_expected'])
print(f'c2_stats_dnn {c2_stats_dnn}')
c2_stats_dnn_d = stats.chisquare(f_obs=dnn_d_gb_s1.dropna()['actual'], f_exp=dnn_d_gb_s1.dropna()['count_expected'])
print(f'c2_stats_dnn_d {c2_stats_dnn_d}')

pfa_gb_s1.to_csv(os.path.join(result_dir, f'pfa_gb_s1.csv'), index=False)
pfa_gb_s1.to_csv(os.path.join(result_dir, f'pfa_gb_s1.csv'), index=False)
pfa_gb_s1.to_csv(os.path.join(result_dir, f'pfa_gb_s1.csv'), index=False)
pfa_gb_s1.to_csv(os.path.join(result_dir, f'pfa_gb_s1.csv'), index=False)


    # plt.plot(pfa_gb['rate'], pfa_gb['rate'], '-o', label='rate')
    # plt.plot(pfa_gb['rate'], pfa_gb['actual_rate'], '-o', label='pfa')
    # plt.plot(pfa_d_gb['rate'], pfa_d_gb['actual_rate'], '-o', label='pfa_d')
    # plt.plot(dnn_gb['rate'], dnn_gb['actual_rate'], '-o', label='dnn')
    # plt.plot(dnn_d_gb['rate'], dnn_d_gb['actual_rate'], '-o', label='dnn_d')
    # plt.legend(loc='upper left')
    # plt.savefig(os.path.join(result_dir, 'pfa_dnn_dash_compare_all.pdf'), bbox_inches='tight')


# def binning(g):
#     return pd.Series(data={'actual': g.actual.sum(), 'count': len(g.index)})
#
# pfa_dashboard_diff_none_validate_gb = pfa_dashboard_diff_none_validate_stats.groupby(by=['binned_range']).apply(binning).reset_index()
# pfa_dashboard_diff_none_validate_gb['rate'] = pfa_dashboard_diff_none_validate_gb['binned_range'].apply(lambda x: x.right)
# pfa_dashboard_diff_none_validate_gb['expected'] = pfa_dashboard_diff_none_validate_gb['count'] * pfa_dashboard_diff_none_validate_gb['rate']
# pfa_dashboard_diff_none_validate_gb['actual_rate'] = pfa_dashboard_diff_none_validate_gb['actual'] / pfa_dashboard_diff_none_validate_gb['count']
# pfa_dashboard_diff_none_validate_gb.to_csv(os.path.join(run_dir, 'random_forest_validate_outcome_gb.csv'), index=False)
#
# plt.plot(pfa_dashboard_diff_none_validate_gb['rate'], pfa_dashboard_diff_none_validate_gb['actual_rate'], '-o')
# plt.plot(pfa_dashboard_diff_none_validate_gb['rate'], pfa_dashboard_diff_none_validate_gb['rate'], '-o')
# plt.savefig(os.path.join(run_dir, 'random_forest_validate_outcome_gb.pdf'), bbox_inches='tight')



# def add_binning_cols():
#     bins = list(drange_inc(0, 1, '0.05')) # 5% point bin size
#     bin_labels = list(range(1, 21))
#
#
# nn_dashboard_diff_none_validate_stats['binned_ind'] = pd.cut(nn_dashboard_diff_none_validate_stats['probs_1'], bins=bins, labels=bin_labels)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
# nn_dashboard_diff_none_validate_stats['binned_range'] = pd.cut(nn_dashboard_diff_none_validate_stats['probs_1'], bins=bins)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
#
#
# def binning(g):
#     return pd.Series(data={'actual': g.actual.sum(), 'count': len(g.index)})

# TODO total counts for counts
# pfa_vs_dnn.groupby(by="anon_id").

# pred_vs_actual_df.to_csv(os.path.join(run_dir, f'pred_vs_actual.csv'), index=False)
# pred_vs_actual_df.to_csv(os.path.join(run_dir, f'pred_vs_actual.csv'), index=False)


# history_ids_test = pd.read_csv(os.path.join('outputs', f'history_test_l{full_history_length}.csv'))
# pfa_pred_test = pd.read_csv(os.path.join(run_dir_pfa_load, f'pfa_pred_vs_actual_test.csv'))



# df.to_csv(os.path.join(run_dir, f'pfa_pred_vs_actual.csv'), index=False)
#
#
#
# run_dir = os.path.join('runs', f'run_results_pfa')
#
# if not os.path.exists(run_dir):
#     os.makedirs(run_dir)
#
# coef = pfa_coef_counts(pfa_coef())
#
# answer_counts = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_validate_l{full_history_length}.txt'))
#
# df = pd.DataFrame(
#     data=([ac[2][1], pfa_prediction(ac, coef)] for ac in answer_counts),
#     columns=['pfa_cor', 'pfa_pred']
# )
#
# df.to_csv(os.path.join(run_dir, f'pfa_pred_vs_actual.csv'), index=False)

# for ac in answer_counts:
#     current_answer = ac[2][1]
#     prob = pfa_prediction(ac)



# #
# # stdout_add_file(os.path.join(run_dir, 'log.txt'))
# #
# snapshots_validate_embedded = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_embedded_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
# snapshots_validate_labels = pd.io.parsers.read_csv(os.path.join('outputs', f'snapshots_validate_labels_t{model_history_length}_l{lstm_layer_size}_e{lstm_epochs}.csv.gz'), delimiter=",", compression="gzip")
#
# # =========== Load the prediction model ===========
#
# # load model
# # Recreate the exact same model purely from the file
# model = keras.models.load_model(os.path.join(run_dir_load, f'model.h5'))
#
# # =========== Probability version of the model ===========
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#
# # Get predictions from the model
# predictions = probability_model.predict(snapshots_validate_embedded)
# correct_prediction = predictions[:,1].transpose()
#
# # get the actual outcome
# actual_outcome = snapshots_validate_labels.values[:,0].transpose()
#
# # concatenate them together and save to disk
# pred_vs_actual = np.column_stack((correct_prediction, actual_outcome))
# pred_vs_actual_df = pd.DataFrame(data=pred_vs_actual, columns=['prob', 'correct'])
# pred_vs_actual_df.to_csv(os.path.join(run_dir, f'pred_vs_actual.csv'), index=False)

# # np.argmax(predictions[0]) # pick the highest chance
# # test_labels[0]
#
#
# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()
