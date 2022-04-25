import numpy as np
import pandas as pd
from Labels import TESTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
import sklearn.metrics as metrics
import os
import pickle
from sklearn.impute import KNNImputer


def hyper_tuning(model, params, X, y, log_path="./data/subtask1_cvresults.csv", saveEstimator=True):
    # GridSearch
    clf = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=10, verbose=3, refit=True)
    clf.fit(X, y)
    # Get cv results
    cv_results = pd.DataFrame().from_dict(clf.cv_results_)
    cv_results.to_csv(log_path)
    # Make prediction by best_estimator on whole train data
    final_model = clf.best_estimator_
    pred = final_model.predict_proba(X)
    pred_ = np.array(pred)[:, :, 1].transpose()
    task1 = metrics.roc_auc_score(y, pred_)
    print(f"Overall Train roc_auc: {task1}")
    if saveEstimator:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        pickle.dump(final_model, open("./log/subtask1_best.p", "wb"))
    return final_model


def train(model, X, y):
    # CV
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

        y_pred = np.transpose([pred[:, 1] for pred in y_pred])

        score = metrics.roc_auc_score(y_test, y_pred, average='macro')
        print(f"score = {score}")
        scores.append(score)
    print(f"Average cv score: {np.mean(scores)}")

    # Train
    model.fit(X, y)
    return model


def get_features(path="./data", split="train"):
    features = pd.read_csv(os.path.join(path, f'{split}_feature_extracted.csv')).sort_values("pid")
    # Select features
    n_neighbor = 10
    knn_imputer = KNNImputer(n_neighbors=n_neighbor, copy=True)
    X = knn_imputer.fit_transform(features.values)
    y = None
    if split == "train":
        labels = pd.read_csv(os.path.join(path, f'{split}_labels.csv')).sort_values("pid")
        y = labels[TESTS].values
    return X, y


# %%
if __name__ == "__main__":
    # Get features
    X, y = get_features()

    # Set seed
    seed = 2022

    # Model selection
    # Random forest
    rf = RandomForestClassifier(random_state=seed, class_weight="balanced_subsample")
    # CV for Parameter tuning
    params = {
        "n_estimators": [500],
        "max_depth": [15],
    }

    estimator = hyper_tuning(rf, params, X, y, saveEstimator=True)
