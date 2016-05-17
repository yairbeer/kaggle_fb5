from ml_metrics import mapk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import datetime
import automl

"""
Import data
"""
# Optimization parameters
init_classifier_params = {'n_estimators': 20, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto',
                          'min_samples_split': 2, 'n_jobs': 3}
metric_params = {'k':  3}

train = pd.DataFrame.from_csv('train.csv')
print(train)
train_label_col = 'place_id'
train_labels = train[train_label_col]
print(train_labels.value_counts(normalize=True))
print(train_labels.value_counts().shape[0])
del train[train_label_col]

# Remove label column name for test
test = pd.DataFrame.from_csv('test.csv')

# For faster iterations
sub_factor = 10
train = train.iloc[::sub_factor, :]
train_labels = train_labels.iloc[::sub_factor]

train_index = train.index.values
test.index = test.index.values + train.shape[0]
test_index = test.index.values

submission_file = pd.DataFrame.from_csv("sample_submission.csv")

# combing tran and test data
# helps working on all the data and removes factorization problems between train and test
dataframe = pd.concat([train, test], axis=0)

del train
del test

"""
Preprocessing
"""
print('There are %d columns' % train.shape[1])

"""
Base level optimization
"""
sk_opt = automl.AutoSKLearnClassification(df=dataframe, train_index=train_index, test_index=test_index,
                                          train_labels=train_labels, sk_classifier=RandomForestClassifier,
                                          calc_probability=True, metric_fun=mapk, params=init_classifier_params,
                                          n_classes=train_labels.value_counts().shape[0])

"""
Meta level optimization
"""

"""
Final Solution
"""
