import os
import numpy as np
import pandas as pd

from numpy import array


def create_snapshots(history, length=243, groupby=['anon_id']):
    snapshot_history = []

    def group_history(data):
        tmp = data.iloc[:length, :]
        # print("========= tmp =========")
        # print(tmp)
        snapshot_history.extend(history_snapshots(length, tmp, groupby))
        return data

    history.groupby(groupby).apply(group_history)
    return array(snapshot_history)


def history_snapshots(desired_timesteps, user_history, drop=[]):
    # print('history_snapshots')
    # print(user_history)
    history = user_history.drop(columns=drop)

    snapshots = []
    num_events, _ = history.shape
    for i in range(1, num_events+1):
        # print(str(i) + ' of ' + str(num_events))
        history_slice = history.iloc[:i, :]
        # print(history_slice)
        snapshots.append(padded_history(desired_timesteps, history_slice))
    # print(snapshots)
    return array(snapshots)


def padded_history(desired_timesteps, history_slice):
    history_timesteps, feature_num = history_slice.shape

    # depending on which is larger we need to restrict the size of the padded data or the history
    blank_history = max(desired_timesteps - history_timesteps, 0)
    start_history = max(history_timesteps - desired_timesteps, 0)
    padded_shape = (desired_timesteps, feature_num)

    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    padded_history = np.zeros(padded_shape).astype(np.float32)
    # we want to fill in the last elements
    padded_history[blank_history:, :feature_num] = history_slice.iloc[start_history:, :]
    return padded_history
