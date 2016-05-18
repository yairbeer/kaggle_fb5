import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

places = pd.DataFrame.from_csv('places_loc_sqr_weight_2016-05-17-17-36.csv')
train = pd.DataFrame.from_csv('train.csv')

train = pd.merge(left=train, right=places, how='left', left_on='place_id', right_on=places.index.values)
print(train)
train['dist'] = np.sqrt((train['x_x'].values - train['x_y'].values) ** 2 +
                        (train['y_x'].values - train['y_y'].values) ** 2)

plt.hist(np.log(1 + train['dist'].values / train['accuracy'].values), bins=50, range=(0, 1))
plt.show()

np.random.seed(100)
choices = np.random.choice(np.arange(train.shape[0]), 1000, replace=False)

plt.plot(train['accuracy'].values[choices], train['dist'].values[choices], 'ko')
plt.show()
