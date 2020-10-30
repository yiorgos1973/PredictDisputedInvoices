import pandas as pd
from sklearn.metrics import f1_score
import sys

#### Predict Invoice Status Grader ####

def score(actuals, preds):     
    try:
        return f1_score(actuals, preds)
    except Exception as err:
        return "Evaluation failed. Error:{}".format(err)

def read_data(actual_file, submission_file):
    try:
        actual_data=pd.read_csv(actual_file, index_col=0, squeeze=True)
        sub_data=pd.read_csv(submission_file, index_col=0, squeeze=True)       
        return actual_data, sub_data
    except Exception as err:
        raise IOError('Loading failed. Error:{}'.format(err))

try:
    actual_data, submission_data = read_data(sys.argv[1], sys.argv[2])
    actual_data.sort_index(inplace=True)
    submission_data.sort_index(inplace=True)
    assert len(actual_data) == len(submission_data), "Data Length Mismatch"
    assert all(actual_data.index == submission_data.index), "Index Inconsistency"
    score = score(actual_data, submission_data)
    print("Score: {:1.3f}".format(score))
except Exception as err:
    print('Score could not be calculated. Error:', err)