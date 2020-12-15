import pandas as pd
import numpy as np

def questions_details_prep(file):
    questions_details = pd.io.parsers.read_csv(file)
    questions_details = questions_details.fillna(False)
    cols = questions_details.columns.values[2:]
    cols_dict = dict([(c, 'float32') for c in cols])
    questions_details = questions_details.astype(cols_dict)
    questions_details = questions_details.drop(columns=['link', 'tagged', 'MATH'])
    questions_details = questions_details.set_index('question')
    return questions_details


def questions_with_skill(question_data: pd.DataFrame, num_diffs: int, num_skills: int, diff_ind: int, skill_ind: int):
    qd = question_data
    # print(f"on {qd.iloc[:, skill_ind] == 1.}")
    skill_on = qd[qd.iloc[:, skill_ind] == 1.]
    # print(f"off {qd.iloc[:, skill_ind] == 0.}")
    skill_off = qd[qd.iloc[:, skill_ind] == 0.]
    return skill_on, skill_off


def prepend_embedded_to_question_skills(embedded_history: np.array, qd: pd.DataFrame, num_skills: int):
    # embedded_no_skills = np.delete(np.copy(embedded_history), range(embedded_history.size - num_skills, embedded_history.size))
    embedded_no_skills = embedded_history[0: embedded_history.size - num_skills]
    # print(f"embedded_no_skills \n{embedded_no_skills}")


    # # duplicate the embedding in every row
    # embedded_df = pd.DataFrame(data=[embedded_no_skills for r in range(0, len(qd))])
    # # print(f"embedded_df \n{embedded_df}")
    #
    # # then append the
    # return pd.concat([embedded_df, qd], axis=1).to_numpy()
    rows = [np.append(embedded_no_skills, row) for row in qd.to_numpy()]
    return np.array(rows)