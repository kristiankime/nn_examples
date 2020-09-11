import unittest
import os
import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from util.nn import nn_one_hot_skills, nn_dashboard_data, nn_dashboard

from numpy import array
from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal
from unittest.mock import MagicMock

from util.util import padded_history, history_snapshots
from util.util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from util.data import question_history_pd, split_snapshot_history, split_snapshot_history_single, create_embedded_history


class TestDataMethods(unittest.TestCase):

    def test_nn_one_hot_skills__difficulty_specified(self):
        actual = nn_one_hot_skills(num_diffs=2, num_skills=3, diff_ind=0, skill_ind=2)
        expected = array([1., 0.,    0., 0., 1.])
        assert_equal(actual, expected)

    def test_nn_one_hot_skills__has_diff_none_selected(self):
        actual = nn_one_hot_skills(num_diffs=2, num_skills=3, diff_ind=-1, skill_ind=1)
        expected = array([0., 0.,    0., 1., 0.])
        assert_equal(actual, expected)

    def test_nn_one_hot_skills__no_difficulty_levels(self):
        actual = nn_one_hot_skills(num_diffs=0, num_skills=3, diff_ind=-1, skill_ind=1)
        expected = array([0., 1., 0.])
        assert_equal(actual, expected)

    def test_nn_dashboard_data__works(self):
        actual = nn_dashboard_data(
            # First 3 are embedding last are skills
            embedded_history=array([0.4, 0.5, 0.6, 1., 0., 1.]),
            num_diffs=0,
            num_skills=3,
            diff_ind=-1)

        expected = array([
            [0.4, 0.5, 0.6, 1., 0., 0.],
            [0.4, 0.5, 0.6, 0., 1., 0.],
            [0.4, 0.5, 0.6, 0., 0., 1.],
        ])
        assert_equal(actual, expected)

    # def test_pfa_dashboard__using_diff(self):
    #     coef = pfa_coef_counts(pd.DataFrame(
    #         columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
    #         data=[
    #             ("diff1", 6., 1.1, -1.1),
    #             ("diff2", 7., 1.1, -1.1),
    #             ("a",    8., 2., -2.),
    #             ("b",    9., 3., -3.),
    #             ("c",   10., 4., -4.),
    #         ]))
    #
    #     data = array(
    #         [[1., 12., 22., 32., 42., 52.],
    #          [0., 11., 21., 31., 41., 51.],
    #          [1.,  1.,  1.,  1.,  1.,  1.]],) # This row shouldn't matter
    #
    #     actual = pfa_dashboard(data_counts=data, coef_counts=coef, num_diffs=2, num_skills=3, diff_ind=1)
    #
    #     expected = array([
    #         pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=0),
    #         pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=1),
    #         pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=2)
    #     ])
    #     assert_equal(actual, expected)
    #
    # def test_print_pfa(self):
    #     coef = pfa_coef_counts(pfa_coef())
    #     data = array(
    #         [[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #          [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #          [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]],) # This row shouldn't matter
    #     dashboard = pfa_dashboard(data_counts=data, coef_counts=coef, num_diffs=0, num_skills=26, diff_ind=-1)
    #     print(dashboard)

