import unittest
import os
import numpy as np
import pandas as pd

from numpy import array
from numpy.testing import assert_array_equal

from util import padded_history

class TestStringMethods(unittest.TestCase):

    def test_foo(self):
        self.assertEqual('foo'.upper(), 'FOO')

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