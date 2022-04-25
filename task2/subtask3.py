import numpy as np

from Labels import VITALS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
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
    seed = 2022
    rf = RandomForestRegressor(random_state=seed)
    params = {
        "n_estimators": [100, 200],
        "max_depth": [9],
        # "criterion": ["mse", "mae"]
    }
    clf = GridSearchCV(rf, param_grid=params, scoring='r2', n_jobs=4, cv=10, verbose=3, refit=True)
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

    norm_score = np.mean(
        [0.5 + 0.5 * np.maximum(0, r2_score(y[:,i], pred[:,i])) for i in range(y.shape[1])])
    print(f"Score normalized $r^2$: {norm_score}")

    if saveEstimator:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        pickle.dump(final_model, open("./log/subtask3_best.p", "wb"))

    return final_model

def train(model, X, y):
    # Manual cross validation
    kf = KFold(n_splits=3, shuffle=True, random_state=1)
    scores = []
    i = 1
    for train_index, test_index in kf.split(X):
        print(f"Fold {i}: ", end='')
        i += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"score = {score}")
        scores.append(score)
    print(f"Average cv score: {np.mean(scores)}")

    # Train
    model.fit(X, y)
    return model

if __name__ == "__main__":
    print("Subtask 3")
    # Get features
    X, y = get_features()

    train3(X, y)
