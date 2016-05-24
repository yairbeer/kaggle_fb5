import pandas as pd
import numpy as np


def dist_xy(ref_xy, cell_xy):
    metric_arr = (ref_xy[0] - cell_xy[:, 0]) ** 2 - (ref_xy[1] - cell_xy[:, 1]) ** 2
    return metric_arr


def find_k_neighbors_index(ref_xy, cell_train_xy, k):
    metric_array = dist_xy(ref_xy, cell_train_xy)
    indexes = np.full((metric_array.shape[0],), fill_value=True, dtype=bool)
    indexes[np.argmin(metric_array[indexes])] = False  # Removing zero is a must
    min_indexes = []
    for i in range(k):
        min_indexes.append(np.argmin(metric_array[indexes]))
        indexes[min_indexes[-1]] = False
    return min_indexes


def knn(ref_xy, cell_train_xy, cell_labels, mapk, knn):
    knn_indexes = find_k_neighbors_index(ref_xy, cell_train_xy, knn)
    knn_result_counts = pd.Series(cell_labels[knn_indexes]).value_counts()
    return knn_result_counts.index.values[:mapk]
