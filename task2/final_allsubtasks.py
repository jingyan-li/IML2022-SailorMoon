#%% md

# IML Task 2

#%%

import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score

#%%

joblib_filename = './log/intermediate_data.joblib'
feature_filename = "./data/train_features.csv"
label_filename = "./data/train_labels.csv"
test_filename = "./data/test_features.csv"
result_filename = "./output/prediction.zip"

#%% md

## Import data

#%%

org_train_feature = pd.read_csv(feature_filename)
org_train_feature.sort_values(by=['pid', 'Time'], inplace=True)

train_label = pd.read_csv(label_filename)
train_label.sort_values(by=['pid'], inplace=True)

org_test_features = pd.read_csv(test_filename)
org_test_features.sort_values(by=['pid', 'Time'], inplace=True)

# LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate,
# LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2.
ORDER_LABEL = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
               'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
               'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
SEPSIS_LABEL = ['LABEL_Sepsis']
SIGN_LABEL = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

print("Data loaded")

#%% md

## Extract features (may takes 5 minutes)

#%%
def trend(y):
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return m
    # return np.polyfit(X, y, 1)[1]


def intercept(y):
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c
    # return np.polyfit(X, y, 1)[0]


feature_agg_funcs = ['min', 'max', 'mean', trend, intercept]
dropping_columns = [('Time', 'min'), ('Time', 'max'), ('Time', 'mean'), ('Time', 'trend'), ('Time', 'intercept'),
                    ('Age', 'min'), ('Age', 'mean'), ('Age', 'trend'), ('Age', 'intercept')]

#%%

# Training data
train_feature = org_train_feature.groupby('pid').agg(feature_agg_funcs)

train_feature.columns = train_feature.columns.to_flat_index()
train_feature.drop(dropping_columns, axis=1, inplace=True)

print("Training feature extracted")
#%%

# Test data
test_feature = org_test_features.groupby('pid').agg(feature_agg_funcs)
test_feature.columns = test_feature.columns.to_flat_index()
test_feature.drop(dropping_columns, axis=1, inplace=True)
print("Test feature extracted")
#%% md
## Fill gaps

#%%
n_neighbor = 10
knn_imputer = KNNImputer(n_neighbors=n_neighbor, copy=True)
imputed_train_feature = knn_imputer.fit_transform(train_feature)

imputed_test_feature = knn_imputer.transform(test_feature)

print("Feature gap filled")
#%% md
## Save and load with `joblib`

#%%

# Save to joblib
with open(joblib_filename, 'wb+') as f:
    pickle_dic = {'train_label': train_label, 'org_train_feature': org_train_feature,
                  'train_feature': train_feature,
                  'knn_imputer': knn_imputer,
                  'imputed_train_feature': imputed_train_feature,
                  'org_test_features': org_test_features, 'test_feature': test_feature,
                  'imputed_test_feature': imputed_test_feature
                  }
    joblib.dump(pickle_dic, f)
    del pickle_dic
print("saved to joblib:" + joblib_filename)

#%%

# Load from joblib
with open(joblib_filename, 'rb') as f:
    pickle_dic = joblib.load(f)
    train_label = pickle_dic['train_label']
    org_train_feature = pickle_dic['org_train_feature']
    train_feature = pickle_dic['train_feature']
    knn_imputer = pickle_dic['knn_imputer']
    imputed_train_feature = pickle_dic['imputed_train_feature']
    org_test_features = pickle_dic['org_test_features']
    test_feature = pickle_dic['test_feature']
    imputed_test_feature = pickle_dic['imputed_test_feature']
print("loaded from joblib:" + joblib_filename)


#%% md
## Subtask 1

#%%
X = imputed_train_feature
y = train_label[ORDER_LABEL].values

n_estimators = 200
n_jobs = -1
clf_1 = MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs), n_jobs=n_jobs)

#%%
# Manual cross validation
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

#%%
# Train
clf_1.fit(X, y)

#%%
# Predict
y_pred_1 = clf_1.predict_proba(imputed_test_feature)
y_pred_1 = np.transpose([pred[:, 1] for pred in y_pred_1])


#%% md
## Subtask 2

#%%

X = imputed_train_feature
y = train_label[SEPSIS_LABEL].values

n_estimators = 300
n_jobs = -1
clf_2 = RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs)

#%%
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

#%%
# Train
clf_2.fit(X, y)

#%%
# Predict
y_pred_2 = clf_2.predict_proba(imputed_test_feature)
y_pred_2 = y_pred_2[:, 1].reshape(-1, 1)

#%% md
## Subtask 3

#%%
X = imputed_train_feature
y = train_label[SIGN_LABEL].values

n_estimators = 200
n_jobs = -1
clf_3 = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs))

#%%
# Manual cross validation
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

#%%
# Train
clf_3.fit(X, y)

#%%
# Predict
y_pred_3 = clf_3.predict(imputed_test_feature)

#%%
output_df = pd.DataFrame(np.concatenate((y_pred_1, y_pred_2, y_pred_3), axis=1).astype(np.float32), index=test_feature.index,
                         columns=ORDER_LABEL+SEPSIS_LABEL+SIGN_LABEL)
org_test_pid = pd.read_csv(test_filename)['pid'].drop_duplicates().values
sorted_output_df = output_df.loc[org_test_pid, :]

sorted_output_df.to_csv(result_filename, compression='zip')