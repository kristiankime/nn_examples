import os
import numpy as np
import pandas as pd

from numpy import array
from util.util import group_snapshots, write_numpy_3d_array_as_txt
from util.data import question_history

history_length = 13
ensure_zeros = 2

# history_length = 243
# ensure_zeros = None

snapshots_train = question_history(os.path.join('outputs', 'answers_history_train.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs', f'snapshot_train_l{history_length}.txt'), fmt='%.0f')

snapshots_validate = question_history(os.path.join('outputs', 'answers_history_validate.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
write_numpy_3d_array_as_txt(snapshots_validate, os.path.join('outputs', f'snapshot_validate_l{history_length}.txt'), fmt='%.0f')

snapshots_test = question_history(os.path.join('outputs', 'answers_history_test.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs', f'snapshot_test_l{history_length}.txt'), fmt='%.0f')

# snapshots_train = question_history(os.path.join('outputs', 'answers_history_train_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs', f'snapshot_train_l{history_length}_10p.txt'), fmt='%.0f')
#
# snapshots_validate = question_history(os.path.join('outputs', 'answers_history_validate_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# write_numpy_3d_array_as_txt(snapshots_validate, os.path.join('outputs', f'snapshot_validate_l{history_length}_10p.txt'), fmt='%.0f')
#
# snapshots_test = question_history(os.path.join('outputs', 'answers_history_test_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs', f'snapshot_test_l{history_length}_10p.txt'), fmt='%.0f')