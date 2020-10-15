import os
import math
import numpy as np
import pandas as pd


def p_m(m):
    return 1 / (1 + math.exp(-1 * m))


def load_pfa_coef():
    return pd.read_csv(os.path.join('data', 'factors_coef_train_1.243diff.csv'))


def pfa_coef_old():
    return pd.DataFrame(
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


def pfa_coef():
    return pd.DataFrame(
        columns=["factor", "intercept", "correct_coef", "incorrect_coef"],
        data=[
            ("very_easy", 1.493481465, 0.031805359, -0.099663404),
            ("easy", 0.816352292, 0.011842884, -0.033759078),
            ("medium", -0.012475342, 0.016069548, -0.018587459),
            ("hard", -0.801974134, 0.022024601, -0.015983519),
            ("very_hard", -1.922273944, 0.074448956, -0.015297359),
            ("Graphing", 0.0513308, 0.016101945, -0.020231609),
            ("Numerical", 0.157490305, 0.022793125, -0.026778728),
            ("Verbal", -0.243549291, 0.109881942, -0.040788796),
            ("Algebraic", 0.262783612, 0.018083795, -0.077606235),
            ("Precalc", 0.029708391, 0.03075907, -0.029056208),
            ("Trig", 0.037269735, 0.00622701, -0.009681269),
            ("Logs", 0.208541782, 0, -0.045451138),
            ("Exp", -0.190838301, 0.015562472, -0.008972252),
            ("Alt.Var.Names", 0.235802927, 0, -0.046748089),
            ("Abstract.Constants", -0.153316235, 0.022900013, 0.004662002),
            ("Limits...Continuity", -0.020073969, 0.013283767, -0.019584892),
            ("Continuity..Definition", 0.356429645, 0, -0.081871871),
            ("Derivative..Definition..Concept", 0.072535364, 0.01317333, -0.017762764),
            ("Derivative..Shortcuts", 0.222130126, 0.00937852, -0.010864698),
            ("Product.Rule", -0.041549313, 0.001418236, -0.021000788),
            ("Quotient.Rule", 0.016943706, 0.015122914, -0.037267658),
            ("Chain.Rule", -0.202470147, 0.007656102, -0.008582573),
            ("Implicit.Differentiation", 0.171533786, 0.004777519, -0.058777138),
            ("Function.Analysis", 0.218745231, 0.00040746, -0.014997761),
            ("Applications", -0.104364945, 0.016050773, -0.006285418),
            ("Antiderivatives", -0.012057895, 0.030996012, -0.008329897),
        ])


def pfa_coef_counts(coef: pd.DataFrame):
    coef = coef.drop(columns=['factor'])
    coef = coef.transpose()
    coef = coef.reindex(["correct_coef", "incorrect_coef", "intercept"])
    coef.insert(0, column="cor", value=[1., 0., 1.])
    return coef.to_numpy()


def pfa_prediction_m(data_counts, coefs):
    current_answer = data_counts[2:3][0][0]
    # print(f'current_answer \n{current_answer}')

    skill_mask = data_counts[2:3][0]
    # print(f'skill_mask \n{skill_mask}')

    # zero columns where the skills are not active
    data_masked = np.multiply(data_counts, skill_mask)
    # print(f'data_masked \n{data_masked}')

    mult = np.multiply(data_masked, coefs)
    # print(f'mult \n{mult}')

    # remove the first column which has the correct / incorrect information that is not used in the PFA computation
    mult_drop_hints = np.delete(mult, 0, axis=1)
    # print(f'mult_drop_hints \n{mult_drop_hints}')

    m = mult_drop_hints.sum()
    # print(f'm \n{m}')

    return m


def pfa_prediction(data_counts, coefs):
    '''
    http://pact.cs.cmu.edu/koedinger/pubs/AIED%202009%20final%20Pavlik%20Cen%20Keodinger%20corrected.pdf
    '''
    return p_m(pfa_prediction_m(data_counts, coefs))


def one_hot_skills(num_diffs: int, num_skills: int, diff_ind: int, skill_ind: int):
    ret = np.zeros(num_diffs + num_skills, dtype=float)
    if (diff_ind >= 0) and (diff_ind < num_diffs):
        ret[diff_ind] = 1.
    ret[num_diffs + skill_ind] = 1.
    return ret


def pfa_dashboard(data_counts, coef_counts, num_diffs: int, num_skills: int, diff_ind: int):
    def pfa_dashboard_skill_n(skill_ind: int):
        # create a "one hot" row with only that skill active (and a hardness?)
        one_hot = one_hot_skills(num_diffs=num_diffs, num_skills=num_skills, diff_ind=diff_ind, skill_ind=skill_ind)
        one_hot = np.concatenate(([0], one_hot))  # hack in a fake "correct" column that is expected below | performance note: https://stackoverflow.com/questions/36998260/prepend-element-to-numpy-array#36998277

        # replace final row with the "one hot"
        data_counts_skill_n = np.copy(data_counts)
        data_counts_skill_n[2] = one_hot

        # run prediction on  that
        skill_n_level = p_m(pfa_prediction_m(data_counts_skill_n, coef_counts))
        return skill_n_level

    # for each skill compute the value int the dashboard
    dashboard = [pfa_dashboard_skill_n(skill_ind) for skill_ind in range(0, num_skills)]
    return np.array(dashboard)


