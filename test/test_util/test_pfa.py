import unittest
import os
import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from util.pfa import p_m, load_pfa_coef, pfa_coef_old, pfa_prediction_m, pfa_coef_counts, one_hot_skills, pfa_dashboard

from numpy import array
from numpy.testing import assert_array_equal, assert_almost_equal, assert_equal
from unittest.mock import MagicMock

from util.util import padded_history, history_snapshots
from util.util import write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from util.data import question_history_pd, split_snapshot_history, split_snapshot_history_single, create_embedded_history


def pred(data, ceof, num_diffs, num_skills, diff_ind, skill_ind):
    d = np.copy(data)
    one_hot = one_hot_skills(num_diffs=num_diffs, num_skills=num_skills, diff_ind=diff_ind, skill_ind=skill_ind)
    one_hot = np.concatenate(([0], one_hot))
    d[2] = one_hot
    return p_m(pfa_prediction_m(d, ceof))


class TestDataMethods(unittest.TestCase):

    def test_pfa_coef_old__works(self):
        actual = pfa_coef_old()

        expected = pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
                ("very_easy", 1.48833883195254, 0.0321045219995053, -0.111218900955991),
                ("easy", 0.803653610932715, 0.0120477213890608, -0.0334294210530982),
                ("medium", -0.0341413947872937, 0.0161322314172315, -0.0179568264766278),
                ("hard", -0.822167560984522, 0.022335669901121, -0.0152368246174216),
                ("very_hard", -1.91424933546937, 0.0685972540846933, -0.0146912492071567),
                ("Graphing", 0.0454331449255232, 0.0154705153840597, -0.0193569010276238),
                ("Numerical", 0.150772463084572, 0.019960151906454, -0.0267971195006408),
                ("Verbal", -0.265303221859572, 0.107936491896628, -0.0392717279374019),
                ("Algebraic", 0.284242010572333, 0.0166242068572127, -0.0764561113903895),
                ("Precalc", 0.0535933907960326, 0.0304591111029889, -0.0291809945867314),
                ("Trig", 0.0310761399460185, 0.00932404373666978, -0.00925948372486153),
                ("Logs", 0.219314096727362, 0, -0.0477568465576301),
                ("Exp", -0.193963275673757, 0.0160091969357439, -0.00923818170498039),
                ("Alt.Var.Names", 0.223383002997616, 0, -0.045956472506932),
                ("Abstract.Constants", -0.133464004552417, 0.0221156084097299, 0.00264342664819196),
                ("Limits...Continuity", 0.00777063924716608, 0.0130919540397894, -0.0219690578203449),
                ("Continuity..Definition", 0.333531518552565, 0, -0.075493341119846),
                ("Derivative..Definition..Concept", 0.0992193717567767, 0.0126078500016887, -0.0186733205580006),
                ("Derivative..Shortcuts", 0.231615425425521, 0.00977653941387463, -0.011381655886295),
                ("Product.Rule", -0.0547747037684012, 0.000984656448090389, -0.0185959830553364),
                ("Quotient.Rule", 0.0079155809679908, 0.0132605911969895, -0.0359905580053593),
                ("Chain.Rule", -0.209628352043356, 0.00652185278266212, -0.00757374396382875),
                ("Implicit.Differentiation", 0.154973689692087, 0.0135214692820953, -0.0646550844186289),
                ("Function.Analysis", 0.216714998864345, 0.00109608688469183, -0.0153387183578819),
                ("Applications", -0.100153816477154, 0.0163144672516737, -0.00663436710196339),
                ("Antiderivatives", 0.0044336240961374, 0.0318700382664901, -0.00985953472874911),
            ])

        assert_frame_equal(actual, expected)

    def test_pfa_coef_counts(self):
        coef = pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
                ("a", 0.11, 1.0, -1.0),
                ("b", 0.22, 2.0, -2.0),
            ])

        actual = pfa_coef_counts(coef)

        expected = array(
            [[1.0,   1.0,   2.0],
             [0.0,  -1.0,  -2.0],
             [1.0,   0.11,  0.22]],)

        assert_equal(actual, expected)

    def test_pfa_prediction_m__works_with_all_skills_active(self):
        coef_pd = pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
            ("a", 9., 1., -1.),
            ("b", 7., 2., -2.),
            ])
        coef = pfa_coef_counts(coef_pd)

        data = array(
            [[1., 12., 22.],
             [0., 11., 21.],
             [1.,  1.,  1.]],)

        actual = pfa_prediction_m(data, coef)

        expected = 9. + (12. * 1.) + (11. * -1.)    +    7. + (22. * 2.) + (21. * -2.)
        assert_equal(actual, expected)

    def test_pfa_prediction_m__works_with_only_some_skills_active(self):
        coef_pd = pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
                ("a", 9., 1., -1.),
                ("b", 7., 2., -2.),
            ])
        coef = pfa_coef_counts(coef_pd)

        data = array(
            [[1., 12., 22.],
             [0., 11., 21.],
             [1.,  1.,  0.]],)

        actual = pfa_prediction_m(data, coef)

        expected = 9. + (12. * 1.) + (11. * -1.)    # This skill is turned off +    7. + (22. * 2.) + (21. * -2.)
        assert_equal(actual, expected)

    def test_one_hot_skills__difficulty_specified(self):
        actual = one_hot_skills(num_diffs=2, num_skills=3, diff_ind=0, skill_ind=2)
        expected = array([1., 0.,    0., 0., 1.])
        assert_equal(actual, expected)

    def test_one_hot_skills__has_diff_none_selected(self):
        actual = one_hot_skills(num_diffs=2, num_skills=3, diff_ind=-1, skill_ind=1)
        expected = array([0., 0.,    0., 1., 0.])
        assert_equal(actual, expected)

    def test_one_hot_skills__no_difficulty_levels(self):
        actual = one_hot_skills(num_diffs=0, num_skills=3, diff_ind=-1, skill_ind=1)
        expected = array([0., 1., 0.])
        assert_equal(actual, expected)

    def test_pfa_dashboard__no_using_diff(self):
        coef = pfa_coef_counts(pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
                ("diff", 6., 1., -1.),
                ("a",    9., 2., -2.),
                ("b",    7., 3., -3.),
            ]))

        data = array(
            [[1., 12., 22., 32],
             [0., 11., 21., 31],
             [1.,  1.,  1., 1.]],) # This row shouldn't matter

        actual = pfa_dashboard(data_counts=data, coef_counts=coef, num_diffs=1, num_skills=2, diff_ind=-1)

        expected = array([
            pred(data, coef, num_diffs=1, num_skills=2, diff_ind=-1, skill_ind=0),
            pred(data, coef, num_diffs=1, num_skills=2, diff_ind=-1, skill_ind=1)
        ])
        assert_equal(actual, expected)

    def test_pfa_dashboard__using_diff(self):
        coef = pfa_coef_counts(pd.DataFrame(
            columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
            data=[
                ("diff1", 6., 1.1, -1.1),
                ("diff2", 7., 1.1, -1.1),
                ("a",    8., 2., -2.),
                ("b",    9., 3., -3.),
                ("c",   10., 4., -4.),
            ]))

        data = array(
            [[1., 12., 22., 32., 42., 52.],
             [0., 11., 21., 31., 41., 51.],
             [1.,  1.,  1.,  1.,  1.,  1.]],) # This row shouldn't matter

        actual = pfa_dashboard(data_counts=data, coef_counts=coef, num_diffs=2, num_skills=3, diff_ind=1)

        expected = array([
            pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=0),
            pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=1),
            pred(data, coef, num_diffs=2, num_skills=3, diff_ind=1, skill_ind=2)
        ])
        assert_equal(actual, expected)

    # def test_print_pfa(self):
    #     coef = pfa_coef_counts(pfa_coef())
    #     data = array(
    #         [[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #          [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #          [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]],) # This row shouldn't matter
    #     dashboard = pfa_dashboard(data_counts=data, coef_counts=coef, num_diffs=0, num_skills=26, diff_ind=-1)
    #     print(dashboard)

