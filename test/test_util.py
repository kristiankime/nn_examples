import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal

from util import padded_history, history_snapshots, create_snapshots

class TestStringMethods(unittest.TestCase):

    def test_foo(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_create_snapshots__default_creates_full_snapshots_for(self):
        data = [[1, 1., 0., 0.,],
                [1, 0., 1., 0.,],
                [1, 0., 0., 1.,],

                [2, 1., 1., 0.,],
                [2, 0., 1., 1.,],
                [2, 1., 0., 1.,],
                ]
        df = pd.DataFrame(data, columns = ['id', 'c1', 'c2', 'c3'])

        result = create_snapshots(df, length=3, groupby=['id'])

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

    def test_history_snapshots__creates_the_number_of_desired_snapshots(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = history_snapshots(3, df)

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

        result = history_snapshots(4, df)

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

        result = history_snapshots(2, df)

        expected = array([[[0., 0., 0.,],
                           [1., 0., 0.,],],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],],
                          [[0., 1., 0.,],
                           [0., 0., 1.,],],
                          ], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_padded_history__history_has_desired_length_does_nothing(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(3, df)

        expected = array([[1., 0., 0.,],
                          [0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)

    def test_padded_history__history_is_lt_desired_length_pads_with_0s(self):
        data = [[1., 0., 0.,],
                [0., 1., 0.,],
                [0., 0., 1.,],]
        df = pd.DataFrame(data, columns = ['c1', 'c2', 'c3'])

        result = padded_history(5, df)

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

        result = padded_history(2, df)

        expected = array([[0., 1., 0.,],
                          [0., 0., 1.,],], dtype=np.float32)
        assert_array_equal(result, expected)