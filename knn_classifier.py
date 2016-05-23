import cProfile
import pandas as pd
import numpy as np


def y2list(y_array):
    y_list = []
    for actual in y_array:
        y_list.append([actual])
    return y_list


def argsort_short(metric_array, k, min_arg=True):
    indexes = np.full((metric_array.shape[0],), fill_value=True, dtype=bool)
    min_indexes = []
    for i in range(k):
        if min_arg:
            min_indexes.append(np.argmin(metric_array[indexes]))
        else:
            min_indexes.append(np.argmax(metric_array[indexes]))
        indexes[min_indexes[-1]] = False
    return min_indexes


def main():
    train = pd.DataFrame.from_csv('train.csv')
    print(train)

    train_label_col = 'place_id'
    train_labels = train[train_label_col]
    label_list = y2list(train_labels)
    del train[train_label_col]

cProfile.run('main()', sort='time')
