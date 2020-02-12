import os
import numpy as np
import pandas as pd

from numpy import array
from util import create_snapshots


answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs' , 'answers_history.csv'))
answer_history_trim = answer_history_base.drop(columns=['question_id', 'timestamp'])
# answer_history_small = answer_history_trim.iloc[:1000,:]

user_size = 100
history_length = 20

users = answer_history_base['anon_id'].unique()
users_n = users[:user_size]

answer_history_n = answer_history_trim[answer_history_trim.anon_id.isin(users_n)]

answer_snapshots = create_snapshots(answer_history_n, length=20)

np.set_printoptions(linewidth=200, threshold=21*20*29)
#np.set_printoptions()
print(answer_snapshots[:21])