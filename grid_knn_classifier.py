import cProfile
import pandas as pd
import numpy as np
import functions_py
from ml_metrics import mapk


def y2list(y_array):
    y_list = y_array.reshape((y_array.shape[0], 1)).tolist()
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


def readfiles():
    """
    Read files
    :return: train and test files
    """
    train_data = pd.DataFrame.from_csv('train.csv')
    test_data = pd.DataFrame.from_csv('test.csv')
    return train_data, test_data


def create_grid(xlim, ylim, step):
    """
    Create ranges for grid calculation
    :param xlim: tuple (min, max)
    :param ylim: tuple (min, max)
    :param step: float
    :return: x,y ranges
    """
    x_range = np.arange(xlim[0], xlim[1], step)
    y_range = np.arange(ylim[0], ylim[1], step)
    return x_range, y_range


def find_index_in_cell(train, test, x_min, y_min, step):
    """
    Find which persons are in a specific cell
    :param train: train in including labels
    :param test: test
    :param x_min: cell's minmimum x
    :param y_min: cell's minmimum y
    :param step: cell width
    :return: train and test persons ARRAY whom are in a specific cell
    """
    data_coor = train['x'].values
    data_in_x = np.logical_and(data_coor > x_min, data_coor < (x_min + step))
    data_coor = train['y'].values
    data_in_y = np.logical_and(data_coor > y_min, data_coor < (y_min + step))
    data_cell = np.logical_and(data_in_x, data_in_y)
    train_cell = train.iloc[data_cell]

    data_coor = test['x'].values
    data_in_x = np.logical_and(data_coor > x_min, data_coor < (x_min + step))
    data_coor = test['y'].values
    data_in_y = np.logical_and(data_coor > y_min, data_coor < (y_min + step))
    data_cell = np.logical_and(data_in_x, data_in_y)
    test_cell = test.iloc[data_cell]

    train_label_col = 'place_id'
    train_labels = train_cell[train_label_col]
    del train_cell[train_label_col]

    return train_cell, test_cell, train_labels

# Work on the train data and test data in the cell
# Split train into train and validation based on time
# KNN each cell


def main():
    step = 0.25
    map_k = 3
    save_name = 'sub_30nn.csv'
    split_val_time = 73000

    train_inc_labels, test = readfiles()
    x_arange, y_arange = create_grid((0, 10), (0, 10), step)

    np.random.seed(2016)
    # train evaluating
    knn_result_list = []
    label_list = []
    for x_cell_min in x_arange:
        for y_cell_min in y_arange:
            print('Working on %f, %f cell' % (x_cell_min + step / 2, y_cell_min + step / 2))
            cur_train, cur_test, cur_labels = find_index_in_cell(train_inc_labels, test, x_cell_min, y_cell_min, step)
            print('Train size is %d, test size is %d' % (cur_train.shape[0], cur_test.shape[0]))
            for i, probe in enumerate(cur_train.values):
                knn_result_list.append(list(functions_py.knn(probe, cur_train.values, cur_labels.values,
                                                             self_test=True, mapk=map_k, k_nn=20)))
                label_list.append([cur_labels.values[i]])
            print('The MAP3 score is %f' % mapk(label_list, knn_result_list, map_k))
            print('***')

    np.random.seed(2016)
    # test predicting
    knn_ids_str = np.full((test.shape[0],), fill_value='0 1 2', dtype=object)
    for x_cell_min in x_arange:
        for y_cell_min in y_arange:
            print('Working on %f, %f cell' % (x_cell_min + step / 2, y_cell_min + step / 2))
            cur_train, cur_test, cur_labels = find_index_in_cell(train_inc_labels, test, x_cell_min, y_cell_min, step)
            print('Train size is %d, test size is %d' % (cur_train.shape[0], cur_test.shape[0]))
            test_index = cur_test.index.values
            for i, probe in enumerate(cur_test.values):
                knn_ids_str[test_index[i]] = ' '.join(list(functions_py.knn(probe, cur_train.values, cur_labels.values,
                                                                            self_test=False, mapk=map_k,
                                                                            k_nn=20).astype(str)))
                # print(test_index[i], knn_ids_str[test_index[i]])
    submission = pd.DataFrame.from_csv('sample_submission.csv')
    submission['place_id'] = knn_ids_str
    submission.to_csv(save_name)


# cProfile.run('main()', sort='time')
main()
