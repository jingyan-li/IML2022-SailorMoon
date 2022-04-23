import numpy as np

from Labels import VITALS
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import os

def task3():
    train_data =pd.read_csv("./data/traindata_imputed.csv").sort_values(by=['pid'])
    train_feature = pd.read_csv(os.path.join("./data", f'train_feature_extracted.csv')).sort_values("pid").fillna(0)
    X = np.array(train_data.values)
    y = np.array(train_feature[VITALS].values)
    test_data = pd.read_csv(os.path.join("./data", f'test_feature_extracted.csv')).sort_values(by=['pid'])
    test_gro = pd.DataFrame(test_data)
    test_group = test_gro.groupby(by=['pid']).fillna(0)
    imputedtest = pd.DataFrame(test_group)
    imputedtest.to_csv("./data/testdata_imputed.csv")
    testimputed = pd.read_csv("./data/testdata_imputed.csv")
    z = np.array(testimputed.values)
    regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    return regr.predict(z)

if __name__ == "__main__":
    print("Subtask 3")
    #data()
    z =task3()
    dbz = pd.DataFrame(z)
    dbz.to_csv('task3.csv')