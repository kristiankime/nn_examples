import os
import numpy as np
import pandas as pd
import random

np.random.seed(23)  # pandas uses numpy

# Load the user history
answer_history_base = pd.io.parsers.read_csv(os.path.join('outputs', 'answers_history.csv'))

# find all the users
users = answer_history_base['anon_id'].unique()

# split into train, validate test
random.shuffle(users)
num_users = users.size
index_train = int(num_users * .9)
index_validate = int(num_users * .95)
index_test = num_users
users_train = users[:index_train]
users_validate = users[index_train:index_validate]
users_test = users[index_validate:index_test]

# record the actual users just in case
np.savetxt(os.path.join('outputs', 'users_train.txt'), users_train, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_validate.txt'), users_validate, fmt='%d')
np.savetxt(os.path.join('outputs', 'users_test.txt'), users_test, fmt='%d')

# Save the filtered data files
answer_history_train = answer_history_base[answer_history_base['anon_id'].isin(users_train)]
answer_history_validate = answer_history_base[answer_history_base['anon_id'].isin(users_validate)]
answer_history_test = answer_history_base[answer_history_base['anon_id'].isin(users_test)]

answer_history_train.to_csv(os.path.join('outputs' , 'answers_history_train.csv'), index=False)
answer_history_validate.to_csv(os.path.join('outputs' , 'answers_history_validate.csv'), index=False)
answer_history_test.to_csv(os.path.join('outputs' , 'answers_history_test.csv'), index=False)