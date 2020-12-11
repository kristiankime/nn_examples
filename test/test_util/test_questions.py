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

from util.questions import questions_with_skill, prepend_embedded_to_question_skills

# print("expected")
# print(expected)
# print("result")
# print(result)

class TestUtilMethods(unittest.TestCase):

    # ====================== add_binning_cols ======================
    def test_questions_with_skill__split_into_on_and_off(self):
        cols = ['skill1', 'skill2', 'skill3']
        df = pd.DataFrame(columns=cols, data=[
            [0.,  1.,  0.],
            [0.,  1.,  0.],
            [1.,  1.,  0.],
            [1.,  0.,  0.],
            [1.,  0.,  0.],
            [0.,  0.,  0.],
        ])

        (on_result, off_result) = questions_with_skill(df, num_diffs=0, num_skills=len(cols), diff_ind=0, skill_ind=0)

        on_expected = pd.DataFrame(columns=cols, data=[
            [1.,  1.,  0.],
            [1.,  0.,  0.],
            [1.,  0.,  0.],
        ])

        off_expected = pd.DataFrame(columns=cols, data=[
            [0.,  1.,  0.],
            [0.,  1.,  0.],
            [0.,  0.,  0.],
        ])

        assert_array_equal(on_result, on_expected)
        assert_array_equal(off_result, off_expected)

    # ====================== add_binning_cols ======================
    def test_prepend_embedded_to_question_skills__split_into_on_and_off(self):
        cols = ['skill1', 'skill2', 'skill3']
        df = pd.DataFrame(columns=cols, data=[
            [1.,  1.,  0.],
            [1.,  0.,  0.],
            [1.,  0.,  1.],
        ])

        emd = np.array([0.4, 0.5, 0.6, 1.0, 1.0, 1.0])

        result = prepend_embedded_to_question_skills(embedded_history=emd, qd=df, num_skills=3)

        expected = np.array([
            [0.4, 0.5, 0.6, 1.0, 1.0, 0.0],
            [0.4, 0.5, 0.6, 1.0, 0.0, 0.0],
            [0.4, 0.5, 0.6, 1.0, 0.0, 1.0],
        ])

        assert_array_equal(result, expected)
