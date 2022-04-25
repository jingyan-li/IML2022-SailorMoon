import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import os
from Labels import SEPSIS
import pickle
from sklearn.impute import KNNImputer

def get_features(path="./data", split="train"):
    train_features = pd.read_csv(os.path.join(path, f'train_feature_extracted.csv')).sort_values(by=["pid"])
    trainfeaturesgroup = train_features.groupby('pid')
    test_features = pd.read_csv(os.path.join(path, f'test_feature_extracted.csv')).sort_values(by=["pid"])
    testfeaturesgroup = test_features.groupby('pid')
    # Select features
    n_neighbor = 10
    knn_imputer = KNNImputer(n_neighbors=n_neighbor, copy=True)
    X = knn_imputer.fit_transform(train_features.values)
    y = None
    if split == "train":
        labels = pd.read_csv(os.path.join(path, f'{split}_labels.csv')).sort_values("pid")
        y = labels[SEPSIS].values.ravel()
        print(X)
        print(y)
        return X,y
    else:
        Z = knn_imputer.transform(test_features.values)
        print(Z)
        return Z, y

def train2(X, y, log_path="./data/subtask2_cvresults.csv", saveEstimator=True):
    seed = 2022
    rf = RandomForestClassifier(random_state=seed, class_weight="balanced_subsample")
    params = {
        "n_estimators": [500],
        "max_depth": [15],
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

if __name__ == "__main__":
    print("Subtask 2")
    # Get features
    #X, y = get_features()

    #estimator = train2(X, y, saveEstimator=True)

    path = "./data/"
    X2, _ = get_features(path=path, split='test')
    model2 = pickle.load(open('./log/subtask2_best.p', "rb"))
    # Predict
    pred2 = model2.predict(X2) # of shape [n_samples, n_classes]
    #pred2 = model2.predict(X2)[:, 1:]
    print("subtask2 done!")
    print(pred2)
    df_true = pd.read_csv("sample.csv").sort_values("pid")
    df_true = df_true.sort_values('pid')
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], pred2['LABEL_Sepsis'])
    print(task2)
