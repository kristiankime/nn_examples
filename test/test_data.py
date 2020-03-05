import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal

from util import padded_history, history_snapshots, group_snapshots
from util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from data import split_snapshot_history_single

# print("expected")
# print(expected)
# print("result")
# print(result)

class TestDataMethods(unittest.TestCase):
    # ====================== split_snapshot_history_single ======================
    def test_split_snapshot_history_single__works_when_size_is_full_data_set(self):
        data = array([[1., 0., 0.,],
                      [2., 0., 0.,],
                      [3., 0., 0.,],
                      [4., 0., 0.,],
                      [5., 0., 0.,],
                      [6., 1., 1.,],
                      ]
                     , dtype=np.float32)

        (history_split, to_predict_features, to_predict_label) = split_snapshot_history_single(data, 5)

        assert_equal(to_predict_label, 6.)
        assert_array_equal(to_predict_features, array([1., 1.,], dtype=np.float32))

        expected = [
            array([[1., 0., 0.,],
                   [2., 0., 0.,],
                   [3., 0., 0.,],
                   [4., 0., 0.,],
                   [5., 0., 0.,],], dtype=np.float32),
        ]
        result = history_split
        # print("=== expected ===")
        # print(expected)
        # print("=== result ===")
        # print(result)
        assert_array_equal(history_split, expected)

    def test_split_snapshot_history_single__work_when_size_divides_history_evenly(self):
        data = array([[1., 0., 0.,],
                      [2., 0., 0.,],
                      [3., 0., 0.,],
                      [4., 0., 0.,],
                      [5., 0., 0.,],
                      [6., 0., 0.,],
                      [7., 1., 1.,],
                      ]
                     , dtype=np.float32)

        (history_split, to_predict_features, to_predict_label) = split_snapshot_history_single(data, 2)

        assert_equal(to_predict_label, 7.)
        assert_array_equal(to_predict_features, array([1., 1.,], dtype=np.float32))

        expected = [
            array([[1., 0., 0.,],
                   [2., 0., 0.,],], dtype=np.float32),
            array([[3., 0., 0.,],
                   [4., 0., 0.,],], dtype=np.float32),
            array([[5., 0., 0.,],
                   [6., 0., 0.,],], dtype=np.float32),
        ]
        # result = history_split
        # print("=== expected ===")
        # print(expected)
        # print("=== result ===")
        # print(result)
        assert_array_equal(history_split, expected)

    def test_split_snapshot_history_single__pads_last_part_of_history_if_their_is_a_remainder(self):
        data = array([[1., 0., 0.,],
                      [2., 0., 0.,],
                      [3., 0., 0.,],
                      [4., 0., 0.,],
                      [5., 0., 0.,],
                      [6., 1., 1.,],
                      ]
                     , dtype=np.float32)

        (history_split, to_predict_features, to_predict_label) = split_snapshot_history_single(data, 2)

        assert_equal(to_predict_label, 6.)
        assert_array_equal(to_predict_features, array([1., 1.,], dtype=np.float32))

        expected = [
            array([[0., 0., 0.,],
                   [1., 0., 0.,],], dtype=np.float32),
            array([[2., 0., 0.,],
                   [3., 0., 0.,],], dtype=np.float32),
            array([[4., 0., 0.,],
                   [5., 0., 0.,],], dtype=np.float32),
        ]
        # result = history_split
        # print("=== expected ===")
        # print(expected)
        # print("=== result ===")
        # print(result)
        assert_array_equal(history_split, expected)