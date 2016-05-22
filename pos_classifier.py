import pandas as pd
import numpy as np
import scipy.stats as stats
from ml_metrics import mapk
import cProfile


def y2list(y_array):
    y_list = []
    for actual in y_array:
        y_list.append([actual])
    return y_list


def dist2mapk(predict_dist, k):
    predict_map = []
    ranked_row = list(stats.rankdata(predict_dist, method='ordinal'))
    for op_rank in range(k):
        predict_map.append(ranked_row.index(op_rank + 1))
    return predict_map


def main():
    closest_3places_ids = np.zeros((train.shape[0],)).astype(object)
    map_k = 3
    eff_sigma = 0.00126
    n_rows = train.shape[0]
    print('There are %d rows' % n_rows)
    for i, cur_coor in enumerate(train_coor[:n_rows]):
        if not i % 100:
            print('row %d' % i)
        dist_sqr = (places_x - cur_coor[0]) ** 2 + (places_y - cur_coor[1]) ** 2
        ranked_ids = dist2mapk(dist_sqr, map_k)
        cur_places = []
        for place_id in ranked_ids:
            cur_places.append(places_ID[place_id])
        closest_3places_ids[i] = cur_places
        cur_places_str = ' '.join(map(lambda x: str(x), cur_places))  # For submission

    print('The MAP3 score is %f' % mapk(label_list, list(closest_3places_ids), map_k))

np.set_printoptions(suppress=True)

places = pd.DataFrame.from_csv('places_loc_sqr_weight_2016-05-17-17-36.csv')
places_ID = places.index.values.astype('int64')
places_x = places['x'].values
places_y = places['y'].values

train = pd.DataFrame.from_csv('train.csv')

train_label_col = 'place_id'
train_labels = train[train_label_col]
label_list = y2list(train_labels)
del train[train_label_col]

train_coor = train[['x', 'y']].values

cProfile.run('main()', sort='time')
