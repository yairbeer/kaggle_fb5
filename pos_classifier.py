import pandas as pd
import numpy as np
import scipy.stats as stats
from ml_metrics import mapk


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

np.set_printoptions(suppress=True)
map_k = 10

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
print(places)
print(train)

# closest_3places_ids = np.zeros((train.shape[0],)).astype(object)
closest_3places_ids = []

print('There are %d rows' % train.shape[0])
for i, cur_coor in enumerate(train_coor[: 1000]):
    if not i % 100:
        print('row %d' % i)
    dist_sqr = (places_x - cur_coor[0]) ** 2 + (places_x - cur_coor[1]) ** 2
    ranked_ids = dist2mapk(dist_sqr, map_k)
    cur_places = []
    for place_id in ranked_ids:
        cur_places.append(places_ID[place_id])
    closest_3places_ids.append(cur_places)
    cur_places_str = ' '.join(map(lambda x: str(x), cur_places))  # For submission

print('The MAP3 score is %f' % mapk(label_list[:1000], closest_3places_ids, map_k))
