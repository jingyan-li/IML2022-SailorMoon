import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import pandas as pd
#%%
# Read data
data = np.genfromtxt('./data/train.csv', skip_header=1, delimiter=",")

#%%
y = data[:, 0]
x = data[:, 1:]

#%%



