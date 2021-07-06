import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_frame = pd.read_csv('corn_distance.csv', header=None)
loc = [data_frame.iloc[i,0] for i in range(len(data_frame))]
loc = np.array(loc)

dist = loc[1:] - loc[:-1]
hist, bin_edges = np.histogram(dist, bins=15, range=(2, 10))

plt.hist(dist, bins=17, range=(2,10))
plt.title('Histogram of distance between neighbor corns')
plt.xlabel('Distance (inch)')
plt.ylabel('Number')
plt.show()