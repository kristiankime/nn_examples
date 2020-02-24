import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal

from util import padded_history, history_snapshots, group_snapshots
from util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt


# print("expected")
# print(expected)
# print("result")
# print(result)

class TestUtilMethods(unittest.TestCase):
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

    def test_history_snapshots__ensure_request_number_of_0_snapshots(self):
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

        print("expected")
        print(expected)
        print("result")
        print(result)

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

        write_numpy_3d_array_as_txt(data, '../test_data/test_write_numpy_3d_array_as_txt_and_read_numpy_3d_array_from_txt.txt', fmt='%.1e')

        result = read_numpy_3d_array_from_txt('../test_data/test_write_numpy_3d_array_as_txt_and_read_numpy_3d_array_from_txt.txt')
        assert_array_equal(result, data)
