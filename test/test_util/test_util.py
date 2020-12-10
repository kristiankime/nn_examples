import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal, assert_equal

from util.util import padded_history, history_snapshots, group_snapshots
from util.util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from util.util import pick_1_in_group
from util.util import drange_inc
from util.util import add_binning_cols
from util.util import binned_counts
from util.util import interweave_3_arrays
from util.util import zero_if_single, zero_if_1d


# print("expected")
# print(expected)
# print("result")
# print(result)

class TestUtilMethods(unittest.TestCase):
    # ====================== drange_inc ======================
    def test_drange_inc__works(self):
        result = list(drange_inc(0, .1, '0.05'))
        expected = [0, 0.05, 0.1]
        assert_equal(result, expected)


    # ====================== add_binning_cols ======================
    def test_add_binning_cols__works(self):
        data = [[0.3,],
                [0.6,],
                [1.0,],

                [1.3,],
                [1.6,],
                [2.0,],
                ]
        input = pd.DataFrame(data, columns = ['val'])

        result = add_binning_cols(input, prob_col='val', prefix='pre', bins=list(drange_inc(0, 2, '1')), bin_labels=list(range(1, 3)))

        # print('result')
        # print(result)
        item = result.iloc[0]['pre_range']
        # print('item')
        # print(item)
        # print('type')
        # print(type(item).__name__)

        data = [[0.3,  1, pd.Interval(0.0, 1.0)],
                [0.6,  1, pd.Interval(0.0, 1.0)],
                [1.0,  1, pd.Interval(0.0, 1.0)],
                [1.3,  2, pd.Interval(1.0, 2.0)],
                [1.6,  2, pd.Interval(1.0, 2.0)],
                [2.0,  2, pd.Interval(1.0, 2.0)],
                ]
        expected = pd.DataFrame(data, columns = ['val', 'pre_ind', 'pre_range'])

        assert_array_equal(result, expected)


    # ====================== add_binning_cols ======================
    def test_binned_counts__works(self):
        data = [[0.1,  0, 1, pd.Interval(0.0, 0.5)],
                [0.1,  0, 1, pd.Interval(0.0, 0.5)],
                [0.1,  0, 1, pd.Interval(0.0, 0.5)],
                [0.9,  1, 2, pd.Interval(0.5, 1.0)],
                [0.9,  1, 2, pd.Interval(0.5, 1.0)],
                [0.9,  1, 2, pd.Interval(0.5, 1.0)],
                ]
        df = pd.DataFrame(data, columns = ['val', 'actual', 'pre_ind', 'pre_range'])

        result = binned_counts(df, actual_col='actual', bin_col='pre_range')
        # print(result)

        expected_data = [[pd.Interval(0.0, 0.5),  0,  3, 0.5, 1.5, 0.0],
                         [pd.Interval(0.5, 1.0),  3,  3, 1.0, 3.0, 1.0],
                ]
        expected = pd.DataFrame(expected_data, columns = ['pre_range', 'actual', 'count', 'rate', 'expected', 'actual_rate'])
        assert_array_equal(result, expected)



    # ====================== group_snapshots ======================
    def test_create_snapshots__default_creates_full_snapshots_for_each_row(self):
        data = [[1, 1., 0., 0.,],
                [1, 0., 1., 0.,],
                [1, 0., 0., 1.,],

                [2, 1., 1., 0.,],
                [2, 0., 1., 1.,],
                [2, 1., 0., 1.,],
                ]
        df = pd.DataFrame(data, columns = ['id', 'c1', 'c2', 'c3'])

        result = group_snapshots(df, groupby=['id'])

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],

                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],

                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],


                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 1., 0.,],],

                          [[0., 0., 0.,],
                           [1., 1., 0.,],
                           [0., 1., 1.,],],

                          [[1., 1., 0.,],
                           [0., 1., 1.,],
                           [1., 0., 1.,],],
                          ]
                         , dtype=np.float32)

        assert_array_equal(result, expected)

    def test_create_snapshots__restricted_group_length(self):
        data = [[1, 1., 0., 0.,],
                [1, 0., 1., 0.,],
                [1, 0., 0., 1.,],

                [2, 1., 1., 0.,],
                [2, 0., 1., 1.,],
                [2, 1., 0., 1.,],
                ]
        df = pd.DataFrame(data, columns = ['id', 'c1', 'c2', 'c3'])

        result = group_snapshots(df, groupby=['id'], group_slice=slice(None, 2), snapshot_length=3)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],

                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],


                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 1., 0.,],],

                          [[0., 0., 0.,],
                           [1., 1., 0.,],
                           [0., 1., 1.,],],
                          ]
                         , dtype=np.float32)

        assert_array_equal(result, expected)

    def test_create_snapshots__ensure_snapshot_length(self):
        data = [[1, 1., 0., 0.,],
                [1, 0., 1., 0.,],
                [1, 0., 0., 1.,],

                [2, 1., 1., 0.,],
                [2, 0., 1., 1.,],
                [2, 1., 0., 1.,],
                ]
        df = pd.DataFrame(data, columns = ['id', 'c1', 'c2', 'c3'])

        result = group_snapshots(df, groupby=['id'], snapshot_length=4)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],

                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],

                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],


                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 1., 0.,],],

                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 1., 0.,],
                           [0., 1., 1.,],],

                          [[0., 0., 0.,],
                           [1., 1., 0.,],
                           [0., 1., 1.,],
                           [1., 0., 1.,],],
                          ]
                         , dtype=np.float32)

        assert_array_equal(result, expected)

    # ====================== history_snapshots ======================
    def test_history_snapshots__snapshot_length_defaults_to_history_length(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        assert_array_equal(result, expected)

    def test_history_snapshots__creates_the_number_of_desired_snapshots(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        assert_array_equal(result, expected)

    def test_history_snapshots__pads_extra_if_requested(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 4)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        assert_array_equal(result, expected)

    def test_history_snapshots__truncates_history_if_requested(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 2)

        expected = array([[[0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_history_snapshots__normally_leaves_all_0_snapshots(self):
        data = [[0., 0., 0.,],
                [1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],],
                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        assert_array_equal(result, expected)

    def test_history_snapshots__skips_all_0_snapshots_if_requested(self):
        data = [[0., 0., 0.,],
                [1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3, ensure_zeroes=0)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        assert_array_equal(result, expected)

    def test_history_snapshots__check_the_request_number_of_all_0_snapshots_is_returned(self):
        data = [[0., 0., 0.,],
                [1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3, ensure_zeroes=2)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],],
                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],],
                          ], dtype=np.float32)

        # print("expected")
        # print(expected)
        # print("result")
        # print(result)

        assert_array_equal(result, expected)

    def test_history_snapshots__check_we_can_ensure_the_number_of_all_0_snapshots_is_one(self):
        data = [[0., 0., 0.,],
                [1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(df, 3, ensure_zeroes=1)

        expected = array([[[0., 0., 0.,],
                           [0., 0., 0.,],
                           [0., 0., 0.,],],
                          [[0., 0., 0.,],
                           [0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[0., 0., 0.,],
                           [1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)

        # print("expected")
        # print(expected)
        # print("result")
        # print(result)

        assert_array_equal(result, expected)

    # ====================== padded_history ======================
    def test_padded_history__length_defaults_to_history_length(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(df, 3)

        expected = array([[1., 0., 0.,],
                          [0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_padded_history__history_has_desired_length_does_nothing(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(df, 3)

        expected = array([[1., 0., 0.,],
                          [0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_padded_history__history_is_lt_desired_length_pads_with_0s(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(df, 5)

        expected = array([[0., 0., 0.,],
                          [0., 0., 0.,],
                          [1., 0., 0.,],
                          [0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_padded_history__history_is_gt_desired_length_strips_history(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(df, 2)

        expected = array([[0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)

    # ====================== write_numpy_3d_array_as_txt and read_numpy_3d_array_from_txt ======================
    def test_write_numpy_3d_array_as_txt_and_read_numpy_3d_array_from_txt(self):
        data = array([[[0., 0., 0.,],
                       [1., 0., 0.,],],
                      [[1., 0., 0.,],
                       [0., 1., 0.,],],
                      [[0., 1., 0.,],
                       [0., 0., 1.,],],
                      ], dtype=np.float32)

        import os
        print(f"cwd = {os.getcwd()}")
        write_numpy_3d_array_as_txt(data, '../test_data/test_write_numpy_3d_array_as_txt_and_read_numpy_3d_array_from_txt.txt', fmt='%.1e')

        result = read_numpy_3d_array_from_txt('../test_data/test_write_numpy_3d_array_as_txt_and_read_numpy_3d_array_from_txt.txt')
        assert_array_equal(result, data)

    # ======== pick_1_in_group =========
    def test_pick_1_in_group(self):
        data = [[1., ],
                [1., ],
                [0., ],
                [0., ]]
        df = pd.DataFrame(data, columns = ['c1'])

        res = pick_1_in_group(df, 'c1')
        # print(res)
        assert True

    # ======== interweave_arrays ======
    def test_interweave_3_arrays(selfs):
        a1 = array([1., 4., 7.,])
        a2 = array([2., 5., 8.,])
        a3 = array([3., 6., 9.,])

        actual = interweave_3_arrays(a1, a2, a3)

        expected = array([1., 2., 3., 4., 5., 6., 7., 8., 9.,])

        # print(f"actual   {actual}")
        # print(f"expected {expected}")
        assert_array_equal(expected, actual)


    # ======== zero_if ======
    def test_zero_if_single__nothing_happens_with_1(self):
        ar = array([1., 2., 3.,])
        actual = zero_if_single(ar, 0)
        expected = array([1., 2., 3.,])
        assert_array_equal(actual, expected)

    def test_zero_if_single__blanks_with_0(self):
        ar = array([0., 2., 3.,])
        actual = zero_if_single(ar, 0)
        expected = array([0., 0., 0.,])
        assert_array_equal(actual, expected)

    def test_zero_if_1d__(self):
        ar = array([1., 1., 3., 4., 0., 5., 6.,])
        actual = zero_if_1d(ar)
        expected = array([1., 1., 3., 4., 0., 0., 0.,])
        assert_array_equal(actual, expected)