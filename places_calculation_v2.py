import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cProfile

__author__ = 'MrBeer'

"""
Import data
"""


def main():
    train = pd.DataFrame.from_csv('train.csv')
    places_index = train['place_id'].values

    places_loc_sqr_wei = []
    for i, place_id in enumerate(train['place_id'].unique()[:10000]):
        place_df = train.iloc[places_index == place_id]
        place_weights_acc_sqred = 1 / (place_df['accuracy'].values ** 2)

        places_loc_sqr_wei.append([place_id, np.average(place_df['x'].values, weights=place_weights_acc_sqred),
                                   np.average(place_df['y'].values, weights=place_weights_acc_sqred),
                                   place_df.shape[0]])

    places_loc_sqr_wei = np.array(places_loc_sqr_wei)

    column_names = ['x', 'y', 'n_persons']
    places_loc_sqr_wei = pd.DataFrame(data=places_loc_sqr_wei[:, 1:], index=places_loc_sqr_wei[:, 0],
                                      columns=column_names)

    # now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # places_loc_sqr_wei.to_csv('places_loc_sqr_weights_%s.csv' % now)
    #
    # plt.plot(places_loc_sqr_wei['x'].values, places_loc_sqr_wei['y'].values, 'ko')
    # plt.show()

cProfile.run('main()', sort='time')
