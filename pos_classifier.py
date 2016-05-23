import pandas as pd
import numpy as np
from ml_metrics import mapk
import cProfile


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


def map_brain(dataset, n_rows, labels=[], save_name=None):

    map_k = 3
    print('There are %d rows' % n_rows)

    data_coor_sigma = dataset[['x', 'y', 'accuracy']].values
    closest_3places_ids = []
    closest_3places_ids_str = np.zeros((dataset.shape[0],)).astype(object)
    for i, cur_coor in enumerate(data_coor_sigma[:n_rows]):
        if not i % 1000:
            print('row %d' % i)
        metric_results = (((places_x - cur_coor[0]) / places_x_sd) ** 2 +
                          ((places_y - cur_coor[1]) / places_y_sd) ** 2) / places_freq
        ranked_ids = argsort_short(metric_results, map_k)
        cur_places = []
        for place_id in ranked_ids:
            cur_places.append(places_ID[place_id])
        closest_3places_ids.append(cur_places)
        closest_3places_ids_str[i] = ' '.join(map(lambda x: str(x), cur_places))  # For submission

    if len(labels):
        print('The MAP3 score is %f' % mapk(labels, closest_3places_ids, map_k))
    if save_name:
        submission = pd.DataFrame.from_csv('sample_submission.csv')
        submission['place_id'] = closest_3places_ids_str
        submission.to_csv(save_name)

np.set_printoptions(suppress=True)

places = pd.DataFrame.from_csv('places_loc_sqr_weights_2016-05-23-13-15.csv')
places.loc[places['n_persons'].values < 20, ['x_sd', 'y_sd']] = 0.001
places_ID = places.index.values.astype('int64')
places_x = places['x'].values
places_x_sd = places['x_sd'].values
places_y = places['y'].values
places_y_sd = places['y_sd'].values
places_freq = places['n_persons'].values
# Fix low count inaccuracy in measurement error
print(places)

train = pd.DataFrame.from_csv('train.csv')
print(train)

train_label_col = 'place_id'
train_labels = train[train_label_col]
label_list = y2list(train_labels)
del train[train_label_col]

eff_sigma = 0.001
cProfile.run('map_brain(train, 30000, labels=label_list)', sort='time')

# del train
#
# test = pd.DataFrame.from_csv('test.csv')
# print(test)
# map_brain(test, test.shape[0], save_name='min_dist_o_freq.csv')

# calculated eff_sigma = 0.00126
# n = 30000
# minimum: dist_sqr = ((places_x - cur_coor[0]) ** 2 + (places_y - cur_coor[1]) ** 2)
# MAP3 = 0.132203
# minimum: dist_sqr = ((places_x - cur_coor[0]) ** 2 + (places_y - cur_coor[1]) ** 2) / places_freq
# MAP3 = 0.163200
# minimum: metric_results = (((places_x - cur_coor[0]) / places_x_sd) ** 2 +
# ((places_y - cur_coor[1]) / places_y_sd) ** 2)
# MAP3 = 0.070661
# minimum: metric_results = (((places_x - cur_coor[0]) / places_x_sd) ** 2 +
# ((places_y - cur_coor[1]) / places_y_sd) ** 2) / places_freq
# MAP3 = 0.127717
# added places.loc[places['n_persons'].values < 20, ['x_sd', 'y_sd']] = 0.001
# MAP3 = 0.
