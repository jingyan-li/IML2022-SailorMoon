import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
import sklearn.metrics as metrics
import os
from Labels import SEPSIS
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
        y = labels[SEPSIS].values.ravel()
    return X, y

def train2(X, y, log_path="./data/subtask2_cvresults.csv", saveEstimator=True):
    seed = 2022
    rf = RandomForestClassifier(random_state=seed, class_weight="balanced_subsample")
    params = {
        "n_estimators": [400, 500, 600],
        "max_depth": [9, 12, 15],
    }
    clf = GridSearchCV(rf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=10, verbose=3, refit=True)
    clf.fit(X, y)
    # Get cv results
    cv_results = pd.DataFrame().from_dict(clf.cv_results_)
    cv_results.to_csv(log_path)
    # Make prediction by best_estimator on whole train data
    final_model = clf.best_estimator_
    pred = final_model.predict_proba(X)
    print(pred.shape)
    pred_ = np.array(pred)[ :, 1].transpose()
    task2 = metrics.roc_auc_score(y, pred_)
    print(f"Overall Train roc_auc: {task2}")

    if saveEstimator:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        pickle.dump(final_model, open("./log/subtask2_best.p", "wb"))
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

        y_pred = model.predict_proba(X_test)
        y_pred = y_pred[:, 1]
        score = metrics.roc_auc_score(y_test, y_pred, average='macro')
        print(f"score = {score}")
        scores.append(score)
    print(f"Average cv score: {np.mean(scores)}")

    # Train
    model.fit(X, y)

    return model

if __name__ == "__main__":
    print("Subtask 2")
    # Get features
    X, y = get_features()

    estimator = train2(X, y, saveEstimator=True)