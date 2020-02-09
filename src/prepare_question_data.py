import os
import numpy as np
import pandas as pd

def questions_links_prep(file):
    # questions_links = pd.io.parsers.read_csv(os.path.join('data' , 'questions_links.csv'))
    questions_links = pd.io.parsers.read_csv(file)
    questions_links = questions_links.loc[:,['problem_source','question_id']].rename(columns={'problem_source': 'question'})
    questions_links = questions_links.set_index('question')
    return questions_links

def questions_details_prep(file):
    questions_details = pd.io.parsers.read_csv(file)
    questions_details = questions_details.fillna(False)
    cols = questions_details.columns.values[2:]
    cols_dict = dict([(c, 'float32') for c in cols])
    questions_details = questions_details.astype(cols_dict)
    questions_details = questions_details.drop(columns=['link'])
    questions_details = questions_details.set_index('question')
    return questions_details


questions_links = questions_links_prep(os.path.join('data' , 'questions_links.csv'))
# questions_links.dtypes

questions_details = questions_details_prep(os.path.join('data' , 'questions_details.243diff.csv'))
# questions_details.dtypes

questions = questions_links.join(questions_details, how='inner')
questions = questions.set_index('question_id')

answers_correct = pd.io.parsers.read_csv(os.path.join('data' , 'answers_correct_attempts_1.csv'))
answers_correct = answers_correct.set_index('question_id')
# answers_correct.dtypes


answers_history = answers_correct.join(questions, how='inner')
# answers_history.dtypes
answers_history.reset_index(level=0, inplace=True)
answers_history = answers_history.sort_values(by=['anon_id', 'timestamp'], ascending=[True, True])
# answers_history.loc[:,['question_id','anon_id','timestamp']]

answers_history.to_csv(os.path.join('outputs' , 'answers_history.csv'), index=False)