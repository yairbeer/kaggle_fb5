import pandas as pd
import numpy as np
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import datetime


def date_parser(df):
    date_recorder = list(map(lambda x: str(x), df['Timestamp'].values))
    df['improved_hour'] = list(map(lambda x: int(x[8:12]), date_recorder))
    # date_recorder = list(map(lambda x: datetime.datetime.strptime(str(x)[:8], '%Y%m%d'), date_recorder))
    # df['yearly_week_recorder'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
    # df['month_recorder'] = list(map(lambda x: int(x.strftime('%m')), date_recorder))
    del df['Timestamp']
    return df


def count_tags(id_tags_cell):
    """
    calculate how many name tags
    :param id_tags_cell: id tags of a single cell
    :return: number of id tags
    """
    if 'null' in id_tags_cell:
        return 0
    else:
        return len(id_tags_cell)


def get_user_tag(df):
    tags_series = list(df['User_Tags'].values)
    tags_series = list(map(lambda x: x.split(','), tags_series))
    # count tags for each column
    tags_count = list(map(lambda x: count_tags(x), tags_series))
    tags_list = []
    for search_item in tags_series:
        for tag in search_item:
            if not tag in tags_list:
                tags_list.append(tag)
    # print('There are %d ID tags' % len(tags_list))
    df_tags = pd.DataFrame(np.zeros((df.shape[0], len(tags_list))), index=df.index, columns=tags_list)
    df_tags_index = df.index.values
    for i, search_item in enumerate(tags_series):
        for tag in search_item:
            df_tags.at[df_tags_index[i], tag] = 1
    df = pd.concat([df, df_tags], axis=1)
    df['n_tags'] = tags_count
    return df


def split_ip(df):
    ip_series = list(df['IP'].values)
    ip_series = list(map(lambda x: x.split('.'), ip_series))
    df['IP0'] = list(map(lambda x: int(x[0]), ip_series))
    df['IP1'] = list(map(lambda x: int(x[1]), ip_series))
    df['IP2'] = list(map(lambda x: int(x[2]), ip_series))
    return df


def col_to_freq(df, col_names):
    for col in col_names:
        print('Changing to frequency %s' %col)
        val_counts = df[col].value_counts()
        df[col + '_freq'] = np.zeros((df.shape[0],))
        for i, val in enumerate(df[col].values):
            df[col + '_freq'].iat[i] = int(val_counts.at[val])
    return df

"""
Import data
"""
# Optimization parameters
param_grid = [
              {
               'silent': [1],
               'nthread': [3],
               'eval_metric': ['auc'],
               'eta': [0.01],
               'objective': ['binary:logistic'],
               'max_depth': [4, 6, 8],
               # 'min_child_weight': [1],
               'num_round': [5000],
               'gamma': [0],
               'subsample': [0.9],
               'colsample_bytree': [0.7],
               'scale_pos_weight': [0.8],
               'n_monte_carlo': [1],
               'cv_n': [4],
               'test_rounds_fac': [1.1],
               'count_n': [0],
               'mc_test': [True]
              }
             ]

train = pd.DataFrame.from_csv('train.csv')
train_label_col = 'coli'
train_labels = train[train_label_col]
print(train_labels.value_counts(normalize=True))
del train[train_label_col]

# Remove label column name for test
test = pd.DataFrame.from_csv('test.csv')

# For faster iterations
sub_factor = 10
train = train.iloc[::sub_factor, :]
train_labels = train_labels.iloc[::sub_factor]

train_index = train.index.values

submission_file = pd.DataFrame.from_csv("sample_submission.csv")

# combing tran and test data
# helps working on all the data and removes factorization problems between train and test
dataframe = pd.concat([train, test], axis=0)

"""
Preprocess
"""

"""
Split into train and test not done apriori in order to save space
"""

print('There are %d columns' % train.shape[1])

"""
Base level optimization
"""
best_score = 0
best_params = 0
best_train_prediction = 0
best_prediction = 0
meta_solvers_train = []
meta_solvers_test = []
best_train = 0
best_test = 0

print('start CV optimization')
mc_round_list = []
mc_acc_mean = []
mc_acc_sd = []
params_list = []
print_results = []
for params in ParameterGrid(param_grid):
    print(params)
    params_list.append(params)
    train_predictions = np.ones((train.shape[0],))
    print('There are %d columns' % train.shape[1])

    # CV
    mc_auc = []
    mc_round = []
    mc_train_pred = []
    # Use monte carlo simulation if needed to find small improvements
    for i_mc in range(params['n_monte_carlo']):
        cv_n = params['cv_n']
        kf = StratifiedKFold(train_labels.values.flatten(), n_folds=cv_n, shuffle=True, random_state=i_mc ** 3)

        # Calculate train predictions over optimized number of rounds
        local_auc = []
        for cv_train_index, cv_test_index in kf:
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            xg_train = xgboost.DMatrix(dataframe.loc[train_index].values[cv_train_index, :], label=y_train)
            xg_test = xgboost.DMatrix(dataframe.loc[train_index].values[cv_test_index, :], label=y_test)

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);

            # predict
            predicted_results = xgclassifier.predict(xg_test)
            train_predictions[cv_test_index] = predicted_results
            local_auc.append(roc_auc_score(y_test, predicted_results))

        print('Accuracy score ', np.mean(local_auc))
        mc_auc.append(np.mean(local_auc))
        mc_train_pred.append(train_predictions)
        mc_round.append(num_round)

    # Getting the mean integer
    mc_train_pred = np.mean(np.array(mc_train_pred), axis=0)

    mc_round_list.append(int(np.mean(mc_round)))
    mc_acc_mean.append(np.mean(mc_auc))
    mc_acc_sd.append(np.std(mc_auc))
    print('The AUC range is: %.5f to %.5f and best n_round: %d' %
          (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print_results.append('The accuracy range is: %.5f to %.5f and best n_round: %d' %
                         (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1], mc_round_list[-1]))
    print('For ', mc_auc)
    print('The AUC of the average prediction is: %.5f' % roc_auc_score(train_labels.values, mc_train_pred))
    meta_solvers_train.append(mc_train_pred)

    # predicting the test set
    if params['mc_test']:
        watchlist = [(xg_train, 'train')]

        num_round = int(mc_round_list[-1] * params['test_rounds_fac'])
        mc_pred = []
        for i_mc in range(params['n_monte_carlo']):
            params['seed'] = i_mc
            xg_train = xgboost.DMatrix(dataframe.loc[train_index], label=train_labels.values.flatten())
            xg_test = xgboost.DMatrix(dataframe.loc[test_index])

            watchlist = [(xg_train, 'train')]

            xgclassifier = xgboost.train(params, xg_train, num_round, watchlist);
            predicted_results = xgclassifier.predict(xg_test)
            mc_pred.append(predicted_results)

        meta_solvers_test.append(np.mean(np.array(mc_pred), axis=0))
        """ Write the last solution (ready for ensemble creation)"""
        print('writing to file')
        mc_train_pred = mc_train_pred
        # print(meta_solvers_test[-1])
        meta_solvers_test[-1] = meta_solvers_test[-1]
        pd.DataFrame(mc_train_pred).to_csv('train_xgboost_opt_dpth%d.csv' % params['max_depth'])
        submission_file['Prediction'] = meta_solvers_test[-1]
        submission_file.to_csv("test_xgboost_opt_dpth%d.csv" % params['max_depth'])

    # saving best score for printing
    if mc_acc_mean[-1] < best_score:
        print('new best log loss')
        best_score = mc_acc_mean[-1]
        best_params = params
        best_train_prediction = mc_train_pred
        if params['mc_test']:
            best_prediction = meta_solvers_test[-1]

print(best_score)
print(best_params)

print(params_list)
print(print_results)
print(mc_acc_mean)
print(mc_acc_sd)

"""
Final Solution
"""
