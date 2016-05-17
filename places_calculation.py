import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

__author__ = 'MrBeer'

"""
Import data
"""
train = pd.DataFrame.from_csv('train.csv')
places = pd.DataFrame(data=train[['x', 'y', 'accuracy']].values, index=train['place_id'].values,
                      columns=['x', 'y', 'accuracy'])
print(train['place_id'].unique().shape)
places_loc_no_wei = []
places_loc_lin_wei = []
places_loc_sqr_wei = []
for i, place_id in enumerate(train['place_id'].unique()):
    place_df = train.iloc[(train['place_id'] == place_id).values]
    place_weights_acc_sqred = 1 / (place_df['accuracy'].values ** 2)
    place_weights_acc = 1 / place_df['accuracy'].values

    places_loc_no_wei.append([place_id, np.average(place_df['x'].values), np.average(place_df['y'].values),
                              place_df.shape[0]])
    places_loc_lin_wei.append([place_id, np.average(place_df['x'].values, weights=place_weights_acc),
                               np.average(place_df['y'].values, weights=place_weights_acc), place_df.shape[0]])
    places_loc_sqr_wei.append([place_id, np.average(place_df['x'].values, weights=place_weights_acc_sqred),
                               np.average(place_df['y'].values, weights=place_weights_acc_sqred), place_df.shape[0]])
    print(i, place_id)

places_loc_no_wei = np.array(places_loc_no_wei)
places_loc_lin_wei = np.array(places_loc_lin_wei)
places_loc_sqr_wei = np.array(places_loc_sqr_wei)

column_names = ['x', 'y', 'n_persons']
places_loc_no_wei = pd.DataFrame(data=places_loc_no_wei[:, 1:], index=places_loc_no_wei[:, 0], columns=column_names)
places_loc_lin_wei = pd.DataFrame(data=places_loc_lin_wei[:, 1:], index=places_loc_lin_wei[:, 0], columns=column_names)
places_loc_sqr_weight = pd.DataFrame(data=places_loc_sqr_wei[:, 1:], index=places_loc_sqr_wei[:, 0],
                                     columns=column_names)

now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
places_loc_no_wei.to_csv('places_loc_no_weights_%s.csv' % now)
places_loc_lin_wei.to_csv('places_loc_lin_weights_%s.csv' % now)
places_loc_sqr_weight.to_csv('places_loc_sqr_weight_%s.csv' % now)

plt.plot(places_loc_sqr_wei['x'].values, places_loc_sqr_wei['y'].values, 'ko')