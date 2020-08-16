import os
import numpy as np
import pandas as pd

from numpy import array
from util.util import pick_1_in_group, group_snapshots, write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from util.data import question_history

np.random.seed(23)

# history_length = 13
# ensure_zeros = 2

# history_length = 25
# ensure_zeros = 2

history_length = 243
ensure_zeros = None

# ==== full size
# not used
# history_ids_train, snapshots_train, answer_counts_train = question_history(os.path.join('outputs', 'answers_history_train.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_train.to_csv(os.path.join('outputs', f'history_train_l{history_length}.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs', f'snapshot_train_l{history_length}.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_train, os.path.join('outputs', f'answer_counts_train_l{history_length}.txt'), fmt='%.0f')


# ======== Pick one snapshot from each student from validation
# Load Data
history_ids_validate = pd.read_csv(os.path.join('outputs', f'history_validate_l{history_length}.csv'))
snapshots_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_validate_l{history_length}.txt'))
answer_counts_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_validate_l{history_length}.txt'))
# get the pick list
history_ids_validate_seq1 = pick_1_in_group(history_ids_validate, 'anon_id', 'seq')
history_ids_validate_seq1.to_csv(os.path.join('outputs', 'pick1', f'history_validate_l{history_length}_seq1.csv'), index=False)
picked = list(history_ids_validate_seq1['seq'])
# use the list to select snapshots
history_ids_validate_s1 = history_ids_validate_seq1[picked]
snapshots_validate_s1 = snapshots_validate[picked]
answer_counts_validate_s1 = answer_counts_validate[picked]
# save snapshots to disk
history_ids_validate_s1.to_csv(os.path.join('outputs', 'pick1', f'history_validate_l{history_length}_s1.csv'), index=False)
write_numpy_3d_array_as_txt(snapshots_validate_s1, os.path.join('outputs', 'pick1', f'snapshot_validate_l{history_length}_s1.txt'), fmt='%.0f')
write_numpy_3d_array_as_txt(answer_counts_validate_s1, os.path.join('outputs', 'pick1', f'answer_counts_validate_l{history_length}_s1.txt'), fmt='%.0f')


# ======== Pick one from each student from test
# history_ids_test, snapshots_test, answer_counts_test = question_history(os.path.join('outputs', 'answers_history_test.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_test.to_csv(os.path.join('outputs', f'history_test_l{history_length}.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs', f'snapshot_test_l{history_length}.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_test, os.path.join('outputs', f'answer_counts_test_l{history_length}.txt'), fmt='%.0f')

# Load Data
history_ids_test = pd.read_csv(os.path.join('outputs', f'history_test_l{history_length}.csv'))
snapshots_test = read_numpy_3d_array_from_txt(os.path.join('outputs', f'snapshot_test_l{history_length}.txt'))
answer_counts_test = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_test_l{history_length}.txt'))
# get the pick list
history_ids_test_seq1 = pick_1_in_group(history_ids_test, 'anon_id', 'seq')
history_ids_test_seq1.to_csv(os.path.join('outputs', 'pick1', f'history_test_l{history_length}_seq1.csv'), index=False)
picked = list(history_ids_test_seq1['seq'])
# use the list to select snapshots
history_ids_test_s1 = history_ids_test_seq1[picked]
snapshots_test_s1 = snapshots_test[picked]
answer_counts_test_s1 = answer_counts_test[picked]
# save snapshots to disk
history_ids_test_s1.to_csv(os.path.join('outputs', 'pick1', f'history_test_l{history_length}_s1.csv'), index=False)
write_numpy_3d_array_as_txt(snapshots_test_s1, os.path.join('outputs', 'pick1', f'snapshot_test_l{history_length}_s1.txt'), fmt='%.0f')
write_numpy_3d_array_as_txt(answer_counts_test_s1, os.path.join('outputs', 'pick1', f'answer_counts_test_l{history_length}_s1.txt'), fmt='%.0f')





# ==== 10% size for rapid iteration work
# history_ids_train, snapshots_train, answer_counts_train = question_history(os.path.join('outputs', 'answers_history_train_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_train.to_csv(os.path.join('outputs', f'history_train_l{history_length}_10p.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs', f'snapshot_train_l{history_length}_10p.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_train, os.path.join('outputs', f'answer_counts_train_l{history_length}_10p.txt'), fmt='%.0f')

# history_ids_validate, snapshots_validate, answer_counts_validate = question_history(os.path.join('outputs', 'answers_history_validate_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_validate.to_csv(os.path.join('outputs', f'history_validate_l{history_length}_10p.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_validate, os.path.join('outputs', f'snapshot_validate_l{history_length}_10p.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_validate, os.path.join('outputs', f'answer_counts_validate_l{history_length}_10p.txt'), fmt='%.0f')

# history_ids_test, snapshots_test, answer_counts_test = question_history(os.path.join('outputs', 'answers_history_test_10p.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_test.to_csv(os.path.join('outputs', f'history_test_l{history_length}_10p.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs', f'snapshot_test_l{history_length}_10p.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_test, os.path.join('outputs', f'answer_counts_test_l{history_length}_10p.txt'), fmt='%.0f')
