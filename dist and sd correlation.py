import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

places = pd.DataFrame.from_csv('places_loc_sqr_weight_2016-05-17-17-36.csv')
print(places)
plt.hist2d(places['x'].values, places['y'].values, bins=200)
plt.show()

train = pd.DataFrame.from_csv('train.csv')

train = pd.merge(left=train, right=places, how='left', left_on='place_id', right_on=places.index.values)
print(train)
train['dist'] = np.sqrt((train['x_x'].values - train['x_y'].values) ** 2 +
                        (train['y_x'].values - train['y_y'].values) ** 2)

hist_data = train['dist'].values / train['accuracy'].values
print(hist_data)
plt.hist(hist_data, bins=100, range=(0, 0.02))
plt.show()
