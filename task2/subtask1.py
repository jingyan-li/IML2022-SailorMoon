import numpy as np
import pandas as pd
from Labels import TESTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import os
import pickle


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
        pickle.dump(final_model, open("./log/subtask1_best.p","wb"))
    return final_model


def get_features(path="./data", split="train"):
    features = pd.read_csv(os.path.join(path, f'{split}_feature_extracted.csv')).sort_values("pid")
    # Select features
    X = features.values
    # Fill nan values
    X = np.nan_to_num(X)
    y = None
    if split=="test":
        labels = pd.read_csv(os.path.join(path, f'{split}_labels.csv')).sort_values("pid")
        y = labels[TESTS].values
    return X, y


#%%
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
        "n_estimators": [400, 500, 600],
        "max_depth": [9, 12, 15],
    }

    estimator = hyper_tuning(rf, params, X, y, saveEstimator=True)


