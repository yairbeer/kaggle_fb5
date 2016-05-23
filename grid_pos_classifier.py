import pandas as pd
import numpy as np


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
    :return: train and test persons whom are in a specific cell
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

    print('There are %d train samples' % train_cell.shape[0])
    print('There are %d test samples' % test_cell.shape[0])
    print(list(train_labels.value_counts())[:100])

    return train_cell, test_cell, train_labels

# Work on the train data and test data in the cell
# Split train into train and validation based on time
# XGBoost each cell

"""
Main code
"""
train_inc_labels, test = readfiles()

step = 1.0
x_arange, y_arange = create_grid((0, 10), (0, 10), step)

for x_cell_min in x_arange:
    for y_cell_min in y_arange:
        print('Working on %d, %d cell' % (x_cell_min + step / 2, y_cell_min + step / 2))
        cur_train, cur_test, cur_labels = find_index_in_cell(train_inc_labels, test, x_cell_min, y_cell_min, step)

