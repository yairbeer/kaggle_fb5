import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cross_validation import StratifiedKFold


class AutoSKLearnClassification:
    """
    Currently only works without probabilities.
    Need to set seed
    """

    def __init__(self, df, train_index, test_index, train_labels, sk_classifier, calc_probability, metric_fun, params,
                 n_montecarlo=1):
        self.df = pd.DataFrame(df)
        self.train_labels = train_labels
        self.train_index = train_index
        self.test_index = test_index
        self.classifier = sk_classifier
        self.metric = metric_fun
        self.prob = calc_probability
        self.mc = n_montecarlo
        self.best_score = 0
        self.best_params = params
        self.best_train_solved = None
        self.best_test_solved = None
        self.author = 'Yair Beer'

    def opt_param(self, param, init_val, min_val, max_val, is_int, iterations):
        cur_params = self.best_params
        cur_params[param] = init_val
        self.sklearn_cv_classification(cur_params)

    def sklearn_cv_classification(self, params):
        # Set current paramer values
        for key, value in params.iteritems():  # Iterating over dictionary
            setattr(self.classifier, key, value)  # setting attribute in a class
        mc_acc_mean = []
        mc_acc_sd = []
        train_predictions = np.ones((self.train_index.shape[0],))

        # CV
        mc_metric = []
        mc_train_pred = []
        # Use monte carlo simulation if needed to find small improvements
        for i_mc in range(self.mc):
            kf = StratifiedKFold(self.train_labels.values.flatten(), n_folds=4, shuffle=True, random_state=i_mc ** 3)

            cur_metric = []
            # Finding optimized number of rounds
            for cv_train_index, cv_test_index in kf:
                y_train = self.train_labels.iloc[cv_train_index].values.flatten()
                y_test = self.train_labels.iloc[cv_test_index].values.flatten()

                # train machine learning
                self.classifier.fit(self.df.loc[self.train_index].values[cv_train_index, :], y_train)

                # predict
                if self.prob:
                    predicted_results = self.classifier.predict_proba(self.df.loc[
                                                                          self.train_index].values[cv_test_index, :])
                else:
                    predicted_results = self.classifier.predict(self.df.loc[
                                                                    self.train_index].values[cv_test_index, :])
                train_predictions[cv_test_index] = predicted_results
                cur_metric.append(self.metric(y_test, predicted_results))

            print('The metric score ', np.mean(cur_metric))
            mc_metric.append(np.mean(cur_metric))
            mc_train_pred.append(train_predictions)

            mc_acc_mean.append(np.mean(mc_metric))
            mc_acc_sd.append(np.std(mc_metric))
            print('The score range is: %.5f to %.5f' %
                  (mc_acc_mean[-1] - mc_acc_sd[-1], mc_acc_mean[-1] + mc_acc_sd[-1]))
            print('For ', mc_metric)
            print('The AUC of the average prediction is: %.5f' % self.metric(self.train_labels.values, mc_train_pred))

            # predicting the test set
            mc_test_pred = []
            for i_mc in range(self.mc):
                self.classifier.fit(self.df.loc[self.train_index], self.train_labels.values.flatten())
                if self.prob:
                    test_predictions = self.classifier.predict_proba(self.df.loc[self.test_index])
                else:
                    test_predictions = self.classifier.predict(self.df.loc[self.test_index])
                mc_test_pred.append(test_predictions)

            if self.mc:
                mc_train_pred = mean_cellwise(mc_train_pred)
                mc_test_pred = mean_cellwise(mc_test_pred)
            else:
                mc_train_pred = mc_train_pred[0]
                mc_test_pred = mc_test_pred[0]

            return mc_train_pred, mc_test_pred


def mean_cellwise(list_arr):
    mean_mat = np.zeros(list_arr[0].shape)
    for arr in list_arr:
        mean_mat += arr
    mean_mat /= len(list_arr)
    return mean_mat

"""
Sub functions to find MAPK
"""


def percent2mapk(predict_percent, k, n_classes):
    predict_map = []
    for i_row, pred_row in enumerate(predict_percent):
        predict_map.append([])
        ranked_row = list(stats.rankdata(pred_row, method='ordinal'))
        for op_rank in range(k):
            predict_map[i_row].append(ranked_row.index(n_classes - op_rank - 1))
    return predict_map


def list2str(predict_list, join_by):
    str_list = []
    for predict_result in predict_list:
        predict_result = list(map(lambda x: str(x), predict_result))
        str_list.append(join_by.join(predict_result))
    return str_list


def y2list(y_array):
    y_list = []
    for actual in y_array:
        y_list.append([actual])
    return y_list