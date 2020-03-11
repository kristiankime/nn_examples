import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal
from unittest.mock import MagicMock

from util import padded_history, history_snapshots, group_snapshots
from util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from data import split_snapshot_history, split_snapshot_history_single, create_embedded_history

# print("expected")
# print(expected)
# print("result")
# print(result)


class FakeModel:
    def __init__(self, predict_function):
        self.predict_function = predict_function

    def predict(self, *args, **kwargs):
        return self.predict_function(*args, **kwargs)


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

    # ====================== create_embedded_history ======================
    def test_create_embedded_history__collapses_each_history_part_to_an_embedding_and_then_appends_them_together(self):
        model = FakeModel(lambda xs: array([x.sum() for x in xs]))
        hist = [
            array([[0., 0., 0.,],
                   [1., 0., 0.,],], dtype=np.float32),
            array([[2., 0., 0.,],
                   [3., 0., 0.,],], dtype=np.float32),
            array([[4., 0., 0.,],
                   [5., 0., 0.,],], dtype=np.float32),
        ]
        final = array([6., 7., 8.,], dtype=np.float32)
        result = create_embedded_history(model, hist, final)
        expected = array([0. + 1.,
                          2. + 3.,
                          4. + 5.,
                          6., 7., 8.], dtype=np.float32)
        assert_array_equal(result, expected)

    # ====================== split_snapshot_history ======================
    def test_split_snapshot_history__collapses_all_snapshots_and_extracts_labels(self):
        model = FakeModel(lambda xs: array([x.sum() for x in xs]))
        data1 = array([[1., 0., 0.,],
                       [2., 0., 0.,],
                       [3., 1., 1.,],
                       ], dtype=np.float32)
        data2 = array([[4., 0., 0.,],
                       [5., 0., 0.,],
                       [6., 1., 1.,],
                       ], dtype=np.float32)
        hist = array([data1, data2])
        result_hist, result_label = split_snapshot_history(model, hist, 2)

        expected_hist = array([[1. + 2., 1., 1.],
                          [4. + 5., 1., 1.],
                          ], dtype=np.float32)

        expected_label = array([3., 6.])

        # print("=== expected hist ===")
        # print(expected_hist)
        # print("=== result hist ===")
        # print(result_hist)
        assert_array_equal(result_hist, expected_hist)

        # print("=== expected label ===")
        # print(expected_label)
        # print("=== result label ===")
        # print(result_label)
        assert_array_equal(result_label, expected_label)

    def test_split_snapshot_history__collapses_all_snapshots_and_extracts_labels_long_history_length(self):
        model = FakeModel(lambda xs: array([x.sum() for x in xs]))
        data1 = array([[1., 0., 0.,],
                       [2., 0., 0.,],
                       [3., 0., 0.,],
                       [4., 0., 0.,],
                       [5., 0., 0.,],
                       [6., 1., 1.,],
                       ], dtype=np.float32)
        data2 = array([[7., 0., 0.,],
                       [8., 0., 0.,],
                       [9., 0., 0.,],
                       [10., 0., 0.,],
                       [11., 0., 0.,],
                       [12., 2., 2.,],
                       ], dtype=np.float32)
        hist = array([data1, data2])
        result_hist, result_label = split_snapshot_history(model, hist, 2)

        expected_hist = array([[0. + 1., 2. + 3., 4. + 5., 1., 1.],
                               [0. + 7., 8. + 9., 10. + 11., 2., 2.],
                               ], dtype=np.float32)

        expected_label = array([6., 12.])

        # print("=== expected hist ===")
        # print(expected_hist)
        # print("=== result hist ===")
        # print(result_hist)
        assert_array_equal(result_hist, expected_hist)

        # print("=== expected label ===")
        # print(expected_label)
        # print("=== result label ===")
        # print(result_label)
        assert_array_equal(result_label, expected_label)
