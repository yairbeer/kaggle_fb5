import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

# setattr(a, x, 'Bla')


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
        self.best_score = 0
        self.train_solved = None
        self.test_solved = None
        self.mc = n_montecarlo
        self.classifier_params = params
        self.author = 'Yair Beer'

    def sklearn_cv_classification(self):
        mc_round_list = []
        mc_acc_mean = []
        mc_acc_sd = []
        print_results = []
        train_predictions = np.ones((self.train_index.shape[0],))

        # CV
        mc_auc = []
        mc_round = []
        mc_train_pred = []
        # Use monte carlo simulation if needed to find small improvements
        for i_mc in range(self.mc):
            kf = StratifiedKFold(self.train_labels.values.flatten(), n_folds=4, shuffle=True, random_state=i_mc ** 3)

            local_auc = []
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
                local_auc.append(self.metric(y_test, predicted_results))

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
            print('The AUC of the average prediction is: %.5f' % self.metric(self.train_labels.values, mc_train_pred))

            # predicting the test set
            mc_pred = []
            for i_mc in range(self.mc):
                self.classifier.fit(self.df.loc[self.train_index], self.train_labels.values.flatten())
                predicted_results = self.classifier.predict(self.df.loc[self.test_index])
                mc_pred.append(predicted_results)

            self.test_solved = np.mean(np.array(mc_pred), axis=0)
