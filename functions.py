import datetime
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold


def sklearn_cv_classification(df, train_labels, train_index, test_index, sk_classifier, calc_probability,
                              params, metric_fun, n_montecarlo=1):
    mc_round_list = []
    mc_acc_mean = []
    mc_acc_sd = []
    params_list = []
    print_results = []
    params_list.append(params)
    train_predictions = np.ones((train_index.shape[0],))

    # CV
    mc_auc = []
    mc_round = []
    mc_train_pred = []
    # Use monte carlo simulation if needed to find small improvements
    for i_mc in range(n_montecarlo):
        kf = StratifiedKFold(train_labels.values.flatten(), n_folds=params['cv_n'], shuffle=True,
                             random_state=i_mc ** 3)

        local_auc = []
        # Finding optimized number of rounds
        for cv_train_index, cv_test_index in kf:
            y_train = train_labels.iloc[cv_train_index].values.flatten()
            y_test = train_labels.iloc[cv_test_index].values.flatten()

            # train machine learning
            sk_classifier.fit(df.loc[train_index].values[cv_train_index, :], y_train)

            # predict
            if calc_probability:
                predicted_results = sk_classifier.predict_proba(df.loc[train_index].values[cv_test_index, :])
            else:
                predicted_results = sk_classifier.predict(df.loc[train_index].values[cv_test_index, :])
            train_predictions[cv_test_index] = predicted_results
            local_auc.append(metric_fun(y_test, predicted_results))

        print('Accuracy score ', np.mean(local_auc))
        mc_auc.append(np.mean(local_auc))
        mc_train_pred.append(train_predictions)

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
        print('The AUC of the average prediction is: %.5f' % metric_fun(train_labels.values, mc_train_pred))
        meta_solvers_train.append(mc_train_pred)

        # predicting the test set
        if params['mc_test']:
            mc_pred = []
            for i_mc in range(params['n_monte_carlo']):
                params['seed'] = i_mc
                sk_classifier.fit(df.loc[train_index], train_labels.values.flatten())
                predicted_results = sk_classifier.predict(df.loc[test_index])
                mc_pred.append(predicted_results)

            meta_solvers_test.append(np.mean(np.array(mc_pred), axis=0))
    return 0


def date_parser(df, date_col_name, date_str):
    """
    Date parse
    :param df: dataframe
    :param date_col_name: date column name
    :param date_str: date string
    :return: dataframe with parsed date
    """
    date_recorder = list(map(lambda x: str(x), df[date_col_name].values))
    date_recorder = list(map(lambda x: datetime.datetime.strptime(x, date_str), date_recorder))
    df['yearly_week'] = list(map(lambda x: int(x.strftime('%W')), date_recorder))
    del df['date_col_name']
    return df


def col_to_freq(df, col_names):
    for col in col_names:
        print('Changing to frequency %s' %col)
        val_counts = df[col].value_counts()
        df[col + '_freq'] = np.zeros((df.shape[0],))
        for i, val in enumerate(df[col].values):
            df[col + '_freq'].iat[i] = int(val_counts.at[val])
    return df


def count_tags(count_list, null_val=None):
    """
    calculate how many names in each cell
    :param count_list: id tags of a single cell
    :param null_val: null value of zero items
    :return: number of id tags
    """
    if null_val in count_list:
        return 0
    else:
        return len(count_list)


def get_user_tag(df):
    tags_series = list(df['User_Tags'].values)
    tags_series = list(map(lambda x: x.split(','), tags_series))
    # count tags for each column
    tags_count = list(map(lambda x: count_tags(x, 'null'), tags_series))
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
