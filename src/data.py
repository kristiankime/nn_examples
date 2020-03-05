import os
import numpy as np
import pandas as pd

from util import group_snapshots, padded_history

# def question_snapshots(file=os.path.join('outputs' , 'answers_history.csv'), user_size=100, history_length=243):
#     answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs' , 'answers_history.csv'))
#     answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
#
#     users = answer_history_base['anon_id'].unique()
#     users_n = users[:user_size]
#     answer_history_n = answer_history_trim[answer_history_trim.anon_id.isin(users_n)]
#
#     answer_snapshots = group_snapshots(answer_history_n, length=history_length)


# def question_history(file=os.path.join('outputs', 'answers_history.csv'), user_size=None, history_length=243):
#     answer_history_base = pd.io.parsers.read_csv(file)
#     answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
#
#     if user_size is not None:
#         users = answer_history_base['anon_id'].unique()
#         users_n = users[:user_size]
#         answer_history_n = answer_history_trim[answer_history_trim.anon_id.isin(users_n)]
#     else:
#         answer_history_n = answer_history_trim
#
#     answer_snapshots = group_snapshots(answer_history_n, length=history_length)
#     return answer_snapshots

def question_history(file, history_length, ensure_zeros):
    answer_history_base = pd.io.parsers.read_csv(file)
    answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
    answer_snapshots = group_snapshots(answer_history_trim, groupby=['anon_id'], snapshot_length=history_length, ensure_zeroes=ensure_zeros)
    return answer_snapshots


def split_snapshot_history_single(data, size):
    history = data[0:-1]
    history_len = len(history)
    history_remainder = history_len % size

    history_first = history[:history_remainder]
    history_first_padded =  padded_history(history_first, size)

    history_rest = history[history_remainder:]
    history_rest_split = np.split(history_rest, size)
    print(history_rest_split)

    history_split = [history_first_padded] + history_rest_split

    to_predict_all = data[-1:][0]
    to_predict_label = to_predict_all[0]
    to_predict_features = to_predict_all[1:]
    return (history_split, to_predict_features, to_predict_label)

# =========== sample data generation functions
def random_boolean_series(timesteps, feature_num):
    return np.random.random_integers(low=0, high=1, size=(timesteps, feature_num)).astype(np.float32)


def random_boolean_data(samples, timesteps, feature_num):
    return np.array([random_boolean_series(timesteps, feature_num) for i in range(samples)])


def random_history_series(max_timesteps, feature_num):
    history_length = np.random.random_integers(low=1, high=max_timesteps)
    blank_history = max_timesteps - history_length
    filled_shape = (history_length, feature_num)
    padded_shape = (max_timesteps, feature_num)
    filled_history = np.random.random_integers(low=0, high=1, size=filled_shape).astype(np.float32)

    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    padded_history = np.zeros(padded_shape).astype(np.float32)

    # this fills in the first elements where are start of the history, which is not what we want to do
    # padded_history[:filled_shape[0],:filled_shape[1]] = filled_history

    # we want to fill in the last elements
    padded_history[blank_history:, :filled_shape[1]] = filled_history

    return padded_history


def input_history_data(samples, timesteps, feature_num):
    return np.array([random_history_series(timesteps, feature_num) for i in range(samples)])