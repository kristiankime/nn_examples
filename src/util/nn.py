import os
import math
import numpy as np
import pandas as pd


def nn_one_hot_skills(num_diffs: int, num_skills: int, diff_ind: int, skill_ind: int):
    ret = np.zeros(num_diffs + num_skills, dtype=float)
    if (diff_ind >= 0) and (diff_ind < num_diffs):
        ret[diff_ind] = 1.
    ret[num_diffs + skill_ind] = 1.
    return ret


def nn_dashboard_data(embedded_history, num_diffs: int, num_skills: int, diff_ind: int):
    def nn_embedded_with_skill_n(skill_ind: int):
        # switch out the final question information for a "one hot" where one skill is turned on
        one_hot = nn_one_hot_skills(num_diffs=num_diffs, num_skills=num_skills, diff_ind=diff_ind, skill_ind=skill_ind)
        # one_hot = np.concatenate(([0], one_hot))  # hack in a fake "correct" column that is expected below | performance note: https://stackoverflow.com/questions/36998260/prepend-element-to-numpy-array#36998277

        # print(f"one_hot = {one_hot}")
        # replace final questions skills with the "one hot"
        data_counts_skill_n = np.copy(embedded_history)
        data_counts_skill_n[-num_skills:] = one_hot

        # print(f"data_counts_skill_n = {data_counts_skill_n}")

        return data_counts_skill_n

    # for each skill compute the value int the dashboard
    embedded_data = [nn_embedded_with_skill_n(skill_ind) for skill_ind in range(0, num_skills)]
    return np.array(embedded_data)


def nn_dashboard(embedded_history, probability_model, num_diffs: int, num_skills: int, diff_ind: int):
    data = nn_dashboard_data(embedded_history, num_diffs, num_skills, diff_ind)

    # Get predictions from the model
    predictions = probability_model.predict(data)
    correct_prediction = predictions[:,1].transpose()
    return correct_prediction


def nn_dashboard_data_skill_flip(embedded_history, num_diffs: int, num_skills: int, diff_ind: int, flip_value: float):
    def nn_embedded_with_skill_n(skill_ind: int):
        # swap selected final questions skills with the new value
        data_counts_skill_n_flip = np.copy(embedded_history)
        data_counts_skill_n_flip[-(num_skills - skill_ind)] = flip_value

        # print(f"data_counts_skill_n = {data_counts_skill_n_flip}")

        return data_counts_skill_n_flip

    # for each skill compute the value int the dashboard
    embedded_data = [nn_embedded_with_skill_n(skill_ind) for skill_ind in range(0, num_skills)]
    return np.array(embedded_data)


def nn_dashboard_skill_flip(embedded_history, probability_model, num_diffs: int, num_skills: int, diff_ind: int):
    data_flip_on = nn_dashboard_data_skill_flip(embedded_history, num_diffs, num_skills, diff_ind, 1.)
    data_flip_off = nn_dashboard_data_skill_flip(embedded_history, num_diffs, num_skills, diff_ind, 0.)

    # Get predictions when skill is flipped on and off
    predictions_on = probability_model.predict(data_flip_on)
    correct_prediction_on = predictions_on[:, 1].transpose()

    predictions_off = probability_model.predict(data_flip_off)
    correct_prediction_off = predictions_off[:, 1].transpose()

    return correct_prediction_on - correct_prediction_off
