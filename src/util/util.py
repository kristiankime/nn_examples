import numpy as np
import pandas as pd
import re
import itertools
import decimal

from numpy import array


def drange_inc(x, y, jump):
    inc = decimal.Decimal(jump)
    while x <= y:
        yield float(x)
        x += inc


def add_binning_cols(df, prob_col, prefix, bins=list(drange_inc(0, 1, '0.05')), bin_labels=list(range(1, 21))):
    # bins = list(drange_inc(0, 1, '0.05')) # 5% point bin size
    # bin_labels = list(range(1, 21))
    df[f'{prefix}_ind'] = pd.cut(df[prob_col], bins=bins, labels=bin_labels)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
    df[f'{prefix}_range'] = pd.cut(df[prob_col], bins=bins)  # https://stackoverflow.com/questions/45273731/binning-column-with-python-pandas#45273750
    return df


# def binning(g, actual_name='actual', count_name='count'):
#     return pd.Series(data={actual_name: g.actual.sum(), count_name: len(g.index)})


def binned_counts(df, actual_col, bin_col):
    def binning_func(g):
        return pd.Series(data={'actual': g[actual_col].sum(), 'count': len(g.index)})

    df_gb = df.groupby(by=[bin_col]).apply(binning_func).reset_index()
    df_gb['rate'] = df_gb[bin_col].apply(lambda x: x.right)
    df_gb['count_expected'] = df_gb['count'] * df_gb['rate']
    df_gb['actual_rate'] = df_gb['actual'] / df_gb['count']
    return df_gb




def group_snapshots(history, groupby, group_slice=slice(None, None), snapshot_length=None, ensure_zeroes=None):
    snapshot_history = []

    def group_history(data):
        tmp = data.iloc[group_slice, :]
        block = history_snapshots(history=tmp.drop(columns=groupby), desired_timesteps=snapshot_length, ensure_zeroes=ensure_zeroes)
        snapshot_history.extend(block)
        return data

    history.groupby(groupby).apply(group_history)
    return array(snapshot_history)


def history_snapshots(history, desired_timesteps=None, ensure_zeroes=None):
    snapshots = []
    num_events, feature_num = history.shape

    if desired_timesteps is None: # desired_timesteps defaults to the overall number of events
        desired_timesteps = num_events

    zero_count = 0
    for i in range(1, num_events+1):
        history_slice = history.iloc[:i, :]
        history_slice_padded = padded_history(history_slice=history_slice, desired_timesteps=desired_timesteps)

        if ensure_zeroes is not None:
            # If we have a desired amount make sure we keep track of the zero snapshot and add only the desired number
            if np.isin(history_slice_padded, 0.0).all():
                if zero_count < ensure_zeroes:
                    snapshots.append(history_slice_padded)
                zero_count = zero_count + 1
            else:
                snapshots.append(history_slice_padded)
        else:
            # otherwise always add
            snapshots.append(history_slice_padded)

    # If we don't have enough zero snapshots add them
    if (ensure_zeroes is not None) and (zero_count < ensure_zeroes):
        padded_shape = (desired_timesteps, feature_num)
        zero_snapshot = np.zeros(padded_shape).astype(np.float32)
        for _ in itertools.repeat(None, ensure_zeroes - zero_count):
            snapshots.append(zero_snapshot)

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

    if isinstance(history_slice, pd.DataFrame):
        padded_history[blank_history:, :feature_num] = history_slice.iloc[start_history:, :]
    else:
        padded_history[blank_history:, :feature_num] = history_slice[start_history:, :]
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

    # going back to 3D is easy if we know the original shape of the array
    new_data = new_data.reshape((d1, d2, d3))
    return new_data


def pick_1_in_group(df: pd.DataFrame, group_col, pick1col='seq'):
    def callback(x):
        # print('===== x =====')
        # print(x)
        size = len(x)
        ret = [False] * size
        pick = np.random.randint(0, size)
        ret[pick] = True
        # print('===== ret =====')
        # print(ret)
        x[pick1col] = ret
        # print('===== x =====')
        # print(x)
        return x

    # def callback(x):
    #     x['seq'] = range(1, x.shape[0] + 1)
    #     return x
    df = df.groupby(group_col).apply(callback)
    return df
    # def f(x):
    #     size = len(x)
    #     ret = pd.Series(0,index=range(size))
    #     pick = np.random.randint(0, size)
    #     ret[pick] = 1
    #     return ret
    #
    # return data.groupby(group_col).apply(f)