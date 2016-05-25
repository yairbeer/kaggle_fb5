import pandas as pd
import numpy as np


def parse_time(df):
    df['hour'] = (df['time'].values / 60) % 24
    df['weekday'] = (df['time'].values / (60 * 24)) % 7
    return df


def dist_xy(ref_xy, cell_xy):
    metric_arr = (ref_xy[0] - cell_xy[:, 0]) ** 2 + (ref_xy[1] - cell_xy[:, 1]) ** 2
    return metric_arr


def find_k_neighbors_index(ref_xy, cell_train_xy, self_test, k_nn):
    metric_array = dist_xy(ref_xy, cell_train_xy)
    indexes = np.full((metric_array.shape[0],), fill_value=True, dtype=bool)
    if self_test:
        indexes[np.argmin(metric_array[indexes])] = False  # Removing zero is a must
    min_indexes = []
    for i in range(k_nn):
        min_indexes.append(np.argmin(metric_array[indexes]))
        indexes[min_indexes[-1]] = False
    return min_indexes


def knn(ref_xy, cell_train_xy, cell_labels, self_test, mapk, k_nn):
    knn_indexes = find_k_neighbors_index(ref_xy, cell_train_xy, self_test, k_nn)
    knn_result_counts = pd.Series(cell_labels[knn_indexes]).value_counts()
    return knn_result_counts.index.values[:mapk]


def rank_knn():
    return 0


def split_to_val(train_w_lab, time_to_split):
    val_w_lab = train_w_lab.iloc[train_w_lab['time'].values > time_to_split]
    train_w_lab = train_w_lab.iloc[train_w_lab['time'].values <= time_to_split]
    return train_w_lab, val_w_lab
