import pandas as pd
import numpy as np
import datetime

train = pd.DataFrame.from_csv('train.csv')
places = pd.DataFrame.from_csv('places_loc_sqr_weights.csv')

train = pd.merge(left=train, right=places, how='left', left_on='places_id', right_on=places.index.values)
print(train)
