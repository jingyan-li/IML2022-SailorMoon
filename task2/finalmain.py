import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score
from Labels import TESTS, VITALS, SEPSIS

feature_filename = "./data/train_features_extracted.csv"
label_filename = "./data/train_labels.csv"
test_filename = "./data/test_features_extracted.csv"
result_filename = "./prediction.zip"

train_feature = pd.read_csv(feature_filename)
train_feature.sort_values(by=['pid', 'Time'], inplace=True)

train_label = pd.read_csv(label_filename)
train_label.sort_values(by=['pid'], inplace=True)

test_feature = pd.read_csv(test_filename)
test_feature.sort_values(by=['pid', 'Time'], inplace=True)
print("Data loaded")

n_neighbor = 10
knn_imputer = KNNImputer(n_neighbors=n_neighbor, copy=True)
imputed_train_feature = knn_imputer.fit_transform(train_feature)

imputed_test_feature = knn_imputer.transform(test_feature)

print("Data imputed")

## Subtask 1
X = imputed_train_feature
y = train_label[TESTS].values

n_estimators = 200
n_jobs = -1
clf_1 = MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs), n_jobs=n_jobs)

# Cross validation
kf = KFold(n_splits=3, shuffle=True, random_state=1)
scores = []
i = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {i}: ", end = '')
    i += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf_1.fit(X_train, y_train)

    y_pred = clf_1.predict_proba(X_test)

    y_pred = np.transpose([pred[:, 1] for pred in y_pred])

    score = roc_auc_score(y_test, y_pred, average='macro')
    print(f"score = {score}")
    scores.append(score)
print(f"Average cv score: {np.mean(scores)}")

# Train
clf_1.fit(X, y)

# Predict
y_pred_1 = clf_1.predict_proba(imputed_test_feature)
y_pred_1 = np.transpose([pred[:, 1] for pred in y_pred_1])


## Subtask 2
X = imputed_train_feature
y = train_label[SEPSIS].values

n_estimators = 300
n_jobs = -1
clf_2 = RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs)

# Manual cross validation
kf = KFold(n_splits=3, shuffle=True, random_state=1)
scores = []
i = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {i}: ", end = '')
    i += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf_2.fit(X_train, y_train)

    y_pred = clf_2.predict_proba(X_test)
    y_pred = y_pred[:, 1]
    score = roc_auc_score(y_test, y_pred, average='macro')
    print(f"score = {score}")
    scores.append(score)
print(f"Average cv score: {np.mean(scores)}")

# Train
clf_2.fit(X, y)

# Predict
y_pred_2 = clf_2.predict_proba(imputed_test_feature)
y_pred_2 = y_pred_2[:, 1].reshape(-1, 1)

## Subtask 3
X = imputed_train_feature
y = train_label[VITALS].values

n_estimators = 300
n_jobs = -1
clf_3 = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs))

# Cross validation
kf = KFold(n_splits=3, shuffle=True, random_state=1)
scores = []
i = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {i}: ", end = '')
    i += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf_3.fit(X_train, y_train)

    y_pred = clf_3.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"score = {score}")
    scores.append(score)
print(f"Average cv score: {np.mean(scores)}")

# Train
clf_3.fit(X, y)

# Predict
y_pred_3 = clf_3.predict(imputed_test_feature)

output_df = pd.DataFrame(np.concatenate((y_pred_1, y_pred_2, y_pred_3), axis=1).astype(np.float32), index=test_feature.index,
                         columns=TESTS+SEPSIS+VITALS)
org_test_pid = pd.read_csv(test_filename)['pid'].drop_duplicates().values
sorted_output_df = output_df.loc[org_test_pid, :]

sorted_output_df.to_csv(result_filename, compression='zip')