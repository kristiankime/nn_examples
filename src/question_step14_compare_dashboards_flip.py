# https://machinelearningmastery.com/lstm-autoencoders/

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from tensorflow import keras
from numpy import array
from tensorflow.keras.models import Model
from util.logs import stdout_add_file, stdout_reset
from util.util import group_snapshots, read_numpy_3d_array_from_txt
from models import lstm_autoencoder
from util.data import split_snapshot_history_single, split_snapshot_history
from util.nn import nn_dashboard, nn_dashboard_data
from sklearn import preprocessing
from util.dashboard import skills_cols, normalize_dashboard, evaluate_dashboards

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

# pred_model_layer_1 = 1024
# pred_model_layer_2 = 256
# pred_epochs = 80

np.set_printoptions(linewidth=200, threshold=(full_history_length + 1) * model_history_length * feature_num) # unset with np.set_printoptions()

# load dir
load_nn_dir = os.path.join('dashboards', f'nn_dashboard_flip')
load_pfa_dir = os.path.join('dashboards', f'pfa_dashboard')

# model load dir
# nn_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')
# nn_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')

# output location
# run_dir_load = os.path.join('runs', f'run_embedded_l1-{pred_model_layer_1}_l2-{pred_model_layer_2}_e{pred_epochs}')
run_dir = os.path.join('dashboards', f'compare_dashboards_flip')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log.txt'))


# skills_cols = ["very_easy", "easy", "medium", "hard", "very_hard", "Graphing", "Numerical", "Verbal", "Algebraic", "Precalc", "Trig", "Logs", "Exp", "Alt.Var.Names", "Abstract.Constants", "Limits...Continuity", "Continuity..Definition", "Derivative..Definition..Concept", "Derivative..Shortcuts", "Product.Rule", "Quotient.Rule", "Chain.Rule", "Implicit.Differentiation", "Function.Analysis", "Applications", "Antiderivatives"]
# skills_cols = [s.replace(".", "_") for s in skills_cols_raw]
nn_dashboard = pd.read_csv(os.path.join(load_nn_dir, f'nn_dashboard_flip_validate.csv'), header=None).iloc[:,0:26]
nn_dashboard.columns = skills_cols
pfa_dashboard = pd.read_csv(os.path.join(load_pfa_dir, f'pfa_dashboard_diff_none_validate.csv'), header=None).iloc[:,0:26]
pfa_dashboard.columns = skills_cols

nn_dashboard_normalized = normalize_dashboard(nn_dashboard)
# min_max_scaler_nn = preprocessing.MinMaxScaler()
# nn_scaled = min_max_scaler_nn.fit_transform(nn_dashboard.values)
# nn_dashboard_normalized = pd.DataFrame(nn_scaled)
# nn_dashboard_normalized.columns = skills_cols
# min(nn_dashboard_normalized['very_easy'])
# max(nn_dashboard_normalized['very_easy'])
# min(nn_dashboard_normalized['Algebraic'])
# max(nn_dashboard_normalized['Algebraic'])

pfa_dashboard_normalized = normalize_dashboard(pfa_dashboard)
# min_max_scaler_pfa = preprocessing.MinMaxScaler()
# pfa_scaled = min_max_scaler_pfa.fit_transform(pfa_dashboard.values)
# pfa_dashboard_normalized = pd.DataFrame(pfa_scaled)
# pfa_dashboard_normalized.columns = skills_cols
# min(pfa_dashboard_normalized['very_easy'])
# max(pfa_dashboard_normalized['very_easy'])


# def compare_df(df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str, col: str) -> pd.DataFrame:
#     return pd.concat([df1[col], df2[col]], axis=1, keys=[name1 + "_" + col, name2 + "_" + col])
#     # nn_dashboard.iloc[[0]]
#     # pfa_dashboard.iloc[[0]]


# def plot_compare(comp, skill, name):
#     fig1, ax1 = plt.subplots()
#     row_nums = range(0, len(comp.index))
#     ax1.plot(row_nums, comp[f'nn_{skill}'], '-o', label=f'nn_{skill}')
#     ax1.plot(row_nums, comp[f'pfa_{skill}'], '-o', label=f'pfa_{skill}')
#     skill_fn = skill.replace(".", "_")
#     fig1.savefig(os.path.join(run_dir, skill_fn, f'pfa_dnn_dash_compare_{skill_fn}_{name}.pdf'), bbox_inches='tight')

evaluate_dashboards(run_dir, nn_dashboard_normalized, pfa_dashboard_normalized)

# corr_list = []
# # for skill in ['very_easy']:
# for skill in skills_cols:
#
#     comp = compare_df(nn_dashboard_normalized, "nn", pfa_dashboard_normalized, "pfa", skill)
#     comp_1000 = comp.sample(n=1000)
#     comp_sorted_nn = comp.sort_values(by=f'nn_{skill}')
#     comp_sorted_pfa = comp.sort_values(by=f'pfa_{skill}')
#     comp_1000_sorted_nn = comp_1000.sort_values(by=f'nn_{skill}')
#     comp_1000_sorted_pfa = comp_1000.sort_values(by=f'pfa_{skill}')
#
#     skill_fn = skill.replace(".", "_")
#     if not os.path.exists(os.path.join(run_dir, skill_fn)):
#             os.makedirs(os.path.join(run_dir, skill_fn))
#
#     # build a correlation CSV
#     corr = comp.corr().iloc[0][1]
#     corr_list.append((skill, corr))
#
#     # write out the correlation
#     with open(os.path.join(run_dir, skill_fn, f'corr_{skill_fn}.txt'), "w") as text_file:
#         text_file.write(str(comp.corr()))
#
#     plot_compare(comp, skill, "default")
#     plot_compare(comp_sorted_nn, skill, "sorted_nn")
#     plot_compare(comp_sorted_pfa, skill, "sorted_pfa")
#
#     plot_compare(comp_1000, skill, "default_1k")
#     plot_compare(comp_1000_sorted_nn, skill, "sorted_nn_1k")
#     plot_compare(comp_1000_sorted_pfa, skill, "sorted_pfa_1k")
#
# # create DataFrame using data
# corr_df = pd.DataFrame(corr_list, columns =['skill', 'correlation'])
# corr_df.to_csv(os.path.join(run_dir, "correlations.csv"), header=True, index=False)

# # A raw diagram of all the values
    # fig1, ax1 = plt.subplots()
    # row_nums = range(0, len(comp.index))
    # ax1.plot(row_nums, comp[f'nn_{skill}'], '-o', label=f'nn_{skill}')
    # ax1.plot(row_nums, comp[f'pfa_{skill}'], '-o', label=f'pfa_{skill}')
    # fig1.savefig(os.path.join(run_dir, skill, f'pfa_dnn_dash_compare_{skill}.pdf'), bbox_inches='tight')
    #
    # # sorted diagram of all the values
    # fig1, ax1 = plt.subplots()
    # row_nums = range(0, len(comp_sorted.index))
    # ax1.plot(row_nums, comp_sorted[f'nn_{skill}'], '-o', label=f'nn_{skill}')
    # ax1.plot(row_nums, comp_sorted[f'pfa_{skill}'], '-o', label=f'pfa_{skill}')
    # fig1.savefig(os.path.join(run_dir, skill, f'pfa_dnn_dash_compare_{skill}_sorted.pdf'), bbox_inches='tight')



# comp = compare_df(nn_dashboard, "nn", pfa_dashboard, "pfa", "very_easy")
# comp.corr()
#
# fig1, ax1 = plt.subplots()
# row_nums = range(0, len(comp.index))
# ax1.plot(row_nums, comp['nn_very_easy'], '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp['pfa_very_easy'], '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, 'pfa_dnn_dash_compare_very_easy.pdf'), bbox_inches='tight')
#
# comp_sorted = comp.sort_values(by='nn_very_easy')
# fig1, ax1 = plt.subplots()
# row_nums = range(0, len(comp_sorted.index))
# ax1.plot(row_nums, comp_sorted['nn_very_easy'], '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp_sorted['pfa_very_easy'], '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, 'pfa_dnn_dash_compare_very_easy_sorted.pdf'), bbox_inches='tight')
#
#
# fig1, ax1 = plt.subplots()
# num = 2000
# row_nums = range(0, num)
# ax1.plot(row_nums, comp['nn_very_easy'].head(num), '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp['pfa_very_easy'].head(num), '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, f'pfa_dnn_dash_compare_very_easy.{num}.pdf'), bbox_inches='tight')
#
#
# comp_normalized = compare_df(nn_dashboard_normalized, "nn", pfa_dashboard_normalized, "pfa", "very_easy")
# comp_normalized.corr()
#
# fig1, ax1 = plt.subplots()
# row_nums = range(0, len(comp_normalized.index))
# ax1.plot(row_nums, comp_normalized['nn_very_easy'], '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp_normalized['pfa_very_easy'], '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, 'pfa_dnn_dash_compare_normalized_very_easy.pdf'), bbox_inches='tight')
#
# comp_normalized_sorted = comp_normalized.sort_values(by='nn_very_easy')
# fig1, ax1 = plt.subplots()
# row_nums = range(0, len(comp_normalized_sorted.index))
# ax1.plot(row_nums, comp_normalized_sorted['nn_very_easy'], '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp_normalized_sorted['pfa_very_easy'], '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, 'pfa_dnn_dash_compare_normalized_very_easy_sorted.pdf'), bbox_inches='tight')
#
#
# fig1, ax1 = plt.subplots()
# num = 2000
# row_nums = range(0, num)
# ax1.plot(row_nums, comp_normalized['nn_very_easy'].head(num), '-o', label='nn_very_easy')
# ax1.plot(row_nums, comp_normalized['pfa_very_easy'].head(num), '-o', label='pfa_very_easy')
# fig1.savefig(os.path.join(run_dir, f'pfa_dnn_dash_compare_normalized_very_easy.{num}.pdf'), bbox_inches='tight')


# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()

