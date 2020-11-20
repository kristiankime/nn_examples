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
load_nn_dir = os.path.join('dashboards', f'nn_dashboard')
load_pfa_dir = os.path.join('dashboards', f'pfa_dashboard')

# run_dir
run_dir = os.path.join('dashboards', f'compare_dashboards')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

stdout_add_file(os.path.join(run_dir, 'log.txt'))



nn_dashboard = pd.read_csv(os.path.join(load_nn_dir, f'nn_dashboard_diff_none_validate.csv'), header=None).iloc[:,0:26]
nn_dashboard.columns = skills_cols
pfa_dashboard = pd.read_csv(os.path.join(load_pfa_dir, f'pfa_dashboard_diff_none_validate.csv'), header=None).iloc[:,0:26]
pfa_dashboard.columns = skills_cols


nn_dashboard_normalized = normalize_dashboard(nn_dashboard)

# min_max_scaler_nn = preprocessing.MinMaxScaler()
# nn_scaled = min_max_scaler_nn.fit_transform(nn_dashboard.values)
# nn_dashboard_normalized = pd.DataFrame(nn_scaled, columns=skills_cols)
# min(nn_dashboard_normalized['very_easy'])
# max(nn_dashboard_normalized['very_easy'])

pfa_dashboard_normalized = normalize_dashboard(pfa_dashboard)

# min_max_scaler_pfa = preprocessing.MinMaxScaler()
# pfa_scaled = min_max_scaler_pfa.fit_transform(pfa_dashboard.values)
# pfa_dashboard_normalized = pd.DataFrame(pfa_scaled, columns=skills_cols)
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

# for skill in ['very_easy']:
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

# =========== End Reporting ===========
end = datetime.datetime.now()
difference = end - start

print(f'start      {start}')
print(f'end        {end}')
print(f'difference {difference}')

stdout_reset()

