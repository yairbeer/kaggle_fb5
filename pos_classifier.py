import pandas as pd
import numpy as np
from ml_metrics import mapk
import cProfile


def y2list(y_array):
    y_list = []
    for actual in y_array:
        y_list.append([actual])
    return y_list


def argsort_short(metric_array, k):
    # best start
    indexes = np.ones((metric_array.shape[0],)).astype(bool)
    min_indexes = []
    for i in range(k):
        min_indexes.append(np.argmin(metric_array[indexes]))
        indexes[min_indexes[-1]] = False
    return min_indexes


def map_brain(dataset, n_rows, labels=[], save_name=None):

    map_k = 3
    eff_sigma = 0.00126
    print('There are %d rows' % n_rows)

    data_coor = dataset[['x', 'y']].values

    closest_3places_ids = []
    closest_3places_ids_str = np.zeros((dataset.shape[0],)).astype(object)
    for i, cur_coor in enumerate(data_coor[:n_rows]):
        if not i % 100:
            print('row %d' % i)
        dist_sqr = ((places_x - cur_coor[0]) ** 2 + (places_y - cur_coor[1]) ** 2) / places_freq
        ranked_ids = argsort_short(dist_sqr, map_k)
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

places = pd.DataFrame.from_csv('places_loc_sqr_weight_2016-05-17-17-36.csv')
places_ID = places.index.values.astype('int64')
places_x = places['x'].values
places_y = places['y'].values
places_freq = places['n_persons'].values
print(places)

train = pd.DataFrame.from_csv('train.csv')
print(train)

train_label_col = 'place_id'
train_labels = train[train_label_col]
label_list = y2list(train_labels)
del train[train_label_col]

cProfile.run('map_brain(train, 10000, label_list)', sort='time')

del train

test = pd.DataFrame.from_csv('test.csv')
print(test)
map_brain(test, test.shape[0], save_name='min_dist.csv')

# n = 10000
# minimum distance:
# MAP3 = 0.131000
# minimum (distance / n_people)
# MAP3 = 0.173500
