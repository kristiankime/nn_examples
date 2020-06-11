import os
import numpy as np
import pandas as pd
import random

np.random.seed(23)  # pandas uses numpy


def percent_of(percent, total):
    return int(total * percent)


# Load the user history
answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs', 'answers_history.csv'))

# find all the users
users = answer_history_base['anon_id'].unique()

# split into train, validate test
random.shuffle(users)
num_users = users.size
index_train = percent_of(num_users, .90)
index_validate = percent_of(num_users, .95)
index_test = num_users
users_train = users[:index_train]
users_validate = users[index_train:index_validate]
users_test = users[index_validate:index_test]

users_train_10p = users_train[: percent_of(users_train.size, .10)]
users_validate_10p = users_train[: percent_of(users_validate.size, .10)]
users_test_10p = users_train[: percent_of(users_test.size, .10)]

# record the actual users just in case
np.savetxt(os.path.join('outputs', 'users_train.txt'), users_train, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_validate.txt'), users_validate, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_test.txt'), users_test, fmt='%d')

np.savetxt(os.path.join('outputs', 'users_train_10p.txt'), users_train_10p, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_validate_10p.txt'), users_validate_10p, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_test_10p.txt'), users_test_10p, fmt='%d')

# Save the filtered data files
answer_history_train = answer_history_base[answer_history_base['anon_id'].isin(users_train)]
answer_history_validate = answer_history_base[answer_history_base['anon_id'].isin(users_validate)]
answer_history_test = answer_history_base[answer_history_base['anon_id'].isin(users_test)]

answer_history_train.to_csv(os.path.join('outputs', 'answers_history_train.csv'), index=False)
answer_history_validate.to_csv(os.path.join('outputs', 'answers_history_validate.csv'), index=False)
answer_history_test.to_csv(os.path.join('outputs', 'answers_history_test.csv'), index=False)

# 10 percent versions
answer_history_train_10p = answer_history_base[answer_history_base['anon_id'].isin(users_train_10p)]
answer_history_validate_10p = answer_history_base[answer_history_base['anon_id'].isin(users_validate_10p)]
answer_history_test_10p = answer_history_base[answer_history_base['anon_id'].isin(users_test_10p)]

answer_history_train_10p.to_csv(os.path.join('outputs' , 'answers_history_train_10p.csv'), index=False)
answer_history_validate_10p.to_csv(os.path.join('outputs' , 'answers_history_validate_10p.csv'), index=False)
answer_history_test_10p.to_csv(os.path.join('outputs' , 'answers_history_test_10p.csv'), index=False)
