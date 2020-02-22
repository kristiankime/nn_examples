import os
import numpy as np
import pandas as pd
import re

from numpy import array


def group_snapshots(history, groupby=['anon_id'], group_slice=slice(None, None), snapshot_length=None, skip_zeroes=False):
    print("========= history =========")
    print(history)
    snapshot_history = []

    def group_history(data):
        tmp = data.iloc[group_slice, :]

        block = history_snapshots(tmp.drop(columns=groupby), snapshot_length)

        # # if not (skip_zeroes and np.isin(block, 0.0).all()):
        # #     snapshot_history.extend(block)
        snapshot_history.extend(block)
        return data

    history.groupby(groupby).apply(group_history)
    return array(snapshot_history)


def history_snapshots(history, desired_timesteps=None, skip_zeroes=False):
    snapshots = []
    num_events, _ = history.shape

    # desired_timesteps defaults to the overall number of events
    if desired_timesteps is None:
        desired_timesteps = num_events

    for i in range(1, num_events+1):
        history_slice = history.iloc[:i, :]
        history_slice_padded = padded_history(history_slice, desired_timesteps)

        if skip_zeroes and np.isin(history_slice_padded, 0.0).all():
            pass # Here we want to skill all 0 entries so just don't add them
        else:
            snapshots.append(history_slice_padded)

    return array(snapshots)


def padded_history(history_slice, desired_timesteps=None):
    history_timesteps, feature_num = history_slice.shape

    # If desired_timesteps isn't specified use the history length
    if desired_timesteps is None:
        desired_timesteps = history_timesteps

    # depending on which is larger we need to restrict the size of the padded data or the history
    blank_history = max(desired_timesteps - history_timesteps, 0)
    start_history = max(history_timesteps - desired_timesteps, 0)
    padded_shape = (desired_timesteps, feature_num)

    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    padded_history = np.zeros(padded_shape).astype(np.float32)
    # we want to fill in the last elements
    padded_history[blank_history:, :feature_num] = history_slice.iloc[start_history:, :]
    return padded_history


# ======== write numpy array to disk in human readable format
# https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file#3685339

def write_numpy_3d_array_as_txt(data, file='test.txt', fmt='%.18e', delimiter=' ', encoding=None): # header='', footer='', comments='# ',
    # Write the array to disk
    with open(file, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt=fmt, delimiter=delimiter, encoding=encoding)

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


def read_numpy_3d_array_from_txt(file='test.txt'):
    with open(file, 'r') as f:
        first_line = f.readline()
        p = re.compile('# Array shape: \((\d+), (\d+), (\d+)\)')
        result = p.search(first_line)
        d1 = int(result.group(1))
        d2 = int(result.group(2))
        d3 = int(result.group(3))

    # Read the array from disk
    new_data = np.loadtxt(file)

    # However, going back to 3D is easy if we know the
    # original shape of the array
    new_data = new_data.reshape((d1, d2, d3))
    return new_data

