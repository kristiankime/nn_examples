import numpy as np
import pandas as pd
import math

from numpy import array
from util.util import group_snapshots, padded_history, interweave_3_arrays, zero_if_1d

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

def question_history(file, history_length, ensure_zeros) -> (pd.DataFrame, np.ndarray, np.ndarray):
    answer_history_base = pd.io.parsers.read_csv(file)
    return question_history_pd(answer_history_base, history_length, ensure_zeros)


def question_history_pd(answer_history_base, history_length, ensure_zeros) -> (pd.DataFrame, np.ndarray, np.ndarray):
    answer_history_base.sort_values(by=['anon_id', 'timestamp'], ascending=[True, True]) # This should already be done but just in case

    history_ids = answer_history_base[['anon_id', 'question_id', 'timestamp', 'correct']]

    answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
    answer_snapshots = group_snapshots(answer_history_trim, groupby=['anon_id'], snapshot_length=history_length, ensure_zeroes=ensure_zeros)

    def pfa_data(snapshot):
        s = snapshot[0:-1]
        f = snapshot[-1]

        # count correct answers by skill
        c = s[ (1. == s[:,0]) ]
        c = c.sum(axis=0)
        if c.size == 0:
            c = np.zeros(f.size)
        c[0] = 1.

        # count incorrect answers by skill
        i = s[ (0. == s[:,0]) ]
        i = i.sum(axis=0)
        if i.size == 0:
            i = np.zeros(f.size)

        ret = np.array([c, i, f])
        return ret

    answer_counts = np.array([pfa_data(history) for history in answer_snapshots])

    return history_ids, answer_snapshots, answer_counts


def transform_counts_to_thin_format(data):
    def to_thin(counts):
        # print(counts)
        #                          dummy        | correct      | incorrect
        thin = interweave_3_arrays(counts[2][1:], counts[0][1:], counts[1][1:])
        correct_val = counts[2][0]
        # print(f"correct_val = {correct_val}")
        thin = np.insert(thin, 0, correct_val, axis=0)
        return thin

    thin_counts = np.array([to_thin(history) for history in data])
    return thin_counts


def transform_counts_to_thin_format_blank(data):
    thin_counts = transform_counts_to_thin_format(data)
    thin_counts_blank = np.array([zero_if_1d(history) for history in thin_counts])
    return thin_counts_blank


def split_snapshot_history_single(data, size):
    history = data[0:-1]
    history_len = len(history)
    history_remainder = history_len % size

    if history_remainder is not 0:
        history_first = history[:history_remainder]
        history_first_padded = padded_history(history_first, size)

    history_rest = history[history_remainder:]
    history_rest_len = len(history_rest)
    group_size = math.floor(history_rest_len / size)
    history_rest_split = np.split(history_rest, group_size)
    # print(history_rest_split)

    if history_remainder is not 0:
        history_split = [history_first_padded] + history_rest_split
    else:
        history_split = history_rest_split

    to_predict_all = data[-1:][0]
    to_predict_label = to_predict_all[0]
    to_predict_features = to_predict_all[1:]
    return (history_split, to_predict_features, to_predict_label)


def create_embedded_history(model, hist, final):
    # (hist, final, label) = split_snapshot_history_single(snapshot, model_history_length)
    embeddings = model.predict(array(hist))
    embeddings_flat = embeddings.flatten()
    embedded_history = np.append(embeddings_flat, final)
    return embedded_history


def split_snapshot_history(model, histories, size):
    inputs = []
    labels = []
    for history in histories:
        history_split, to_predict_features, to_predict_label = split_snapshot_history_single(history, size)
        embedded_history = create_embedded_history(model, history_split, to_predict_features)
        inputs.append(embedded_history)
        labels.append(to_predict_label)
    return array(inputs), array(labels)

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