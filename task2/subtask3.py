import numpy as np

from Labels import VITALS
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import os
import pickle

def get_features(path="./data", split="train"):
    features = pd.read_csv(os.path.join(path, f'{split}_feature_extracted.csv')).sort_values("pid")
    # Select features
    X = features.values
    # Fill nan values
    X = np.nan_to_num(X)
    y = None
    if split == "train":
        labels = pd.read_csv(os.path.join(path, f'{split}_labels.csv')).sort_values("pid")
        y = labels[VITALS].values
    return X, y


def train3(X, y, log_path="./data/subtask3_cvresults.csv", saveEstimator=True):
    # regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    seed = 2022
    rf = RandomForestRegressor(random_state=seed)
    params = {
        "n_estimators": [400, 500, 600],
        "max_depth": [9, 12, 15],
        "criterion": ["mse", "mae"]
    }
    clf = GridSearchCV(rf, param_grid=params, scoring='r2', n_jobs=-1, cv=10, verbose=3, refit=True)
    clf.fit(X, y)
    # Get cv results
    cv_results = pd.DataFrame().from_dict(clf.cv_results_)
    cv_results.to_csv(log_path)
    # Make prediction by best_estimator on whole train data
    final_model = clf.best_estimator_
    pred = final_model.predict(X)
    print(pred.shape)
    score = r2_score(y, pred)
    print(f"Overall train $r^2$ score : {score}")

    if saveEstimator:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        pickle.dump(final_model, open("./log/subtask3_best.p", "wb"))

    return final_model

if __name__ == "__main__":
    print("Subtask 3")
    # Get features
    X, y = get_features()

    train3(X, y)
