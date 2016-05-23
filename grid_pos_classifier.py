import pandas as pd
import numpy as np


def readfiles():
    """
    Read files
    :return: train and test files
    """
    train_data = pd.DataFrame.from_csv('train.csv')

    train_label_col = 'place_id'
    train_labels = train[train_label_col]
    del train[train_label_col]

    test_data = pd.DataFrame.from_csv('test.csv')
    return train_data, test_data, train_labels


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

# Work on the train data and test data in the cell
# Split train into train and validation based on time
# XGBoost each cell

"""
Main code
"""
train, test, labels = readfiles()

x_arange, y_arange = create_grid((0, 10), (0, 10), 1.0)
