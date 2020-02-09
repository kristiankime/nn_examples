import os
import numpy as np
import pandas as pd

from numpy import array
from util import history_snapshots



# data = array([[1., 0., 0.,],
#               [0., 1., 0.,],
#               [0., 0., 1.,],], dtype=np.float32)

answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs' , 'answers_history.csv'))
answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
answer_history_small = answer_history_trim.iloc[:1000,:]


def create_snapshots(data):
    snapshot_history = []

    def group_history(data):
        tmp = data.iloc[:3,:]
        # print("========= tmp =========")
        # print(tmp)
        snapshot_history.append(history_snapshots(3, tmp, ['anon_id']))
        return data

    answer_history_small.groupby(['anon_id']).apply(group_history)
    return array(snapshot_history)

print(create_snapshots(answer_history_small))