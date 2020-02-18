import os
import numpy as np
import pandas as pd

from numpy import array
from util import create_snapshots, write_numpy_3d_array_as_txt
from data import question_history

history_length = 25

snapshots_train = question_history(os.path.join('outputs', 'answers_history_train.csv'), user_size=None, history_length=history_length)
write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs' , f'snapshot_train_l{history_length}.txt'), fmt='%.1f')

snapshots_validate = question_history(os.path.join('outputs', 'answers_history_validate.csv'), user_size=None, history_length=history_length)
write_numpy_3d_array_as_txt(snapshots_validate, os.path.join('outputs' , f'snapshot_validate_l{history_length}.txt'), fmt='%.1f')

snapshots_test = question_history(os.path.join('outputs', 'answers_history_test.csv'), user_size=None, history_length=history_length)
write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs' , f'snapshot_test_l{history_length}.txt'), fmt='%.1f')