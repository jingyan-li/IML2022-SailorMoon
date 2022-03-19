import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#%%
# Read data
data = np.genfromtxt('./data/train.csv', skip_header=1, delimiter=",")

#%%
# Get data
y = data[:, 0]
x = data[:, 1:]

lmd = [0.1, 1, 10, 100, 200]
SPLITS = 10
#%%
# KFold
kf = KFold(n_splits=SPLITS, shuffle=True)
rmse_arr = []
for l in lmd:
    rmse = 0
    for train_index, test_index in kf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train & fit Ridge regression model
        clf = Ridge(alpha=l, solver='svd',)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        rmse += mean_squared_error(y_test, y_pred, squared=False)
    # Average RMSE over 10 folds
    avg_rmse = rmse / SPLITS
    rmse_arr.append(avg_rmse)
#%%
# Save RMSE
rmse_arr = np.array(rmse_arr)
np.savetxt('submit.csv', rmse_arr, delimiter='\n')



