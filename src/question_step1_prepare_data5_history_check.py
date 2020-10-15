import os
import numpy as np
import pandas as pd

from numpy import array
from util.util import group_snapshots, write_numpy_3d_array_as_txt, read_numpy_3d_array_from_txt
from util.data import question_history, transform_counts_to_thin_format_blank

# history_length = 13
# ensure_zeros = 2

# history_length = 25
# ensure_zeros = 2

history_length = 243
ensure_zeros = None

# ==== full size
# history_ids_train, snapshots_train, answer_counts_train = question_history(os.path.join('outputs', 'answers_history_train.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_train.to_csv(os.path.join('outputs', f'history_train_l{history_length}.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_train, os.path.join('outputs', f'snapshot_train_l{history_length}.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_train, os.path.join('outputs', f'answer_counts_train_l{history_length}.txt'), fmt='%.0f')

# history_ids_validate, snapshots_validate, answer_counts_validate = question_history(os.path.join('outputs', 'answers_history_validate.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_validate.to_csv(os.path.join('outputs', f'history_validate_l{history_length}.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_validate, os.path.join('outputs', f'snapshot_validate_l{history_length}.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_validate, os.path.join('outputs', f'answer_counts_validate_l{history_length}.txt'), fmt='%.0f')
answer_counts_validate = read_numpy_3d_array_from_txt(os.path.join('outputs', f'answer_counts_validate_l{history_length}.txt'))
answer_counts_validate_thin = transform_counts_to_thin_format_blank(answer_counts_validate)
np.savetxt(os.path.join('outputs', f'answer_counts_validate_l{history_length}_thin.csv'), answer_counts_validate_thin, fmt='%.0f', delimiter=",")

# history_ids_test, snapshots_test, answer_counts_test = question_history(os.path.join('outputs', 'answers_history_test.csv'), history_length=history_length, ensure_zeros=ensure_zeros)
# history_ids_test.to_csv(os.path.join('outputs', f'history_test_l{history_length}.csv'), index=False)
# write_numpy_3d_array_as_txt(snapshots_test, os.path.join('outputs', f'snapshot_test_l{history_length}.txt'), fmt='%.0f')
# write_numpy_3d_array_as_txt(answer_counts_test, os.path.join('outputs', f'answer_counts_test_l{history_length}.txt'), fmt='%.0f')



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
