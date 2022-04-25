import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import Labels
import subtask1, subtask2, subtask3
#%%

joblib_filename = './log/intermediate_data.joblib'
feature_filename = "./data/train_features.csv"
label_filename = "./data/train_labels.csv"
test_filename = "./data/test_features.csv"
result_filename = "./output/prediction.zip"


## Import data

org_train_feature = pd.read_csv(feature_filename)
org_train_feature.sort_values(by=['pid', 'Time'], inplace=True)

train_label = pd.read_csv(label_filename)
train_label.sort_values(by=['pid'], inplace=True)

org_test_features = pd.read_csv(test_filename)
org_test_features.sort_values(by=['pid', 'Time'], inplace=True)


ORDER_LABEL = Labels.TESTS
SEPSIS_LABEL = Labels.SEPSIS
SIGN_LABEL = Labels.VITALS

print("Data loaded")


## Extract features


def trend(y):
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return m



def intercept(y):
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c



feature_agg_funcs = ['min', 'max', 'mean', trend, intercept]
dropping_columns = [('Time', 'min'), ('Time', 'max'), ('Time', 'mean'), ('Time', 'trend'), ('Time', 'intercept'),
                    ('Age', 'min'), ('Age', 'mean'), ('Age', 'trend'), ('Age', 'intercept')]


# Training data
train_feature = org_train_feature.groupby('pid').agg(feature_agg_funcs)

train_feature.columns = train_feature.columns.to_flat_index()
train_feature.drop(dropping_columns, axis=1, inplace=True)

print("Training feature extracted")

# Test data
test_feature = org_test_features.groupby('pid').agg(feature_agg_funcs)
test_feature.columns = test_feature.columns.to_flat_index()
test_feature.drop(dropping_columns, axis=1, inplace=True)
print("Test feature extracted")

# Interpolation
n_neighbor = 10
knn_imputer = KNNImputer(n_neighbors=n_neighbor, copy=True)
imputed_train_feature = knn_imputer.fit_transform(train_feature)
imputed_test_feature = knn_imputer.transform(test_feature)

print("Feature gap filled")




## Subtask 1
print("Subtask 1")
X = imputed_train_feature
y = train_label[ORDER_LABEL].values

n_estimators = 200
n_jobs = -1
clf_1 = MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs), n_jobs=n_jobs)

clf_1 = subtask1.train(clf_1, X, y)

# Predict
y_pred_1 = clf_1.predict_proba(imputed_test_feature)
y_pred_1 = np.transpose([pred[:, 1] for pred in y_pred_1])



## Subtask 2

print("Subtask 2")

X = imputed_train_feature
y = train_label[SEPSIS_LABEL].values

n_estimators = 300
n_jobs = -1
clf_2 = RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs)

clf_2 = subtask2.train(clf_2, X, y)

# Predict
y_pred_2 = clf_2.predict_proba(imputed_test_feature)
y_pred_2 = y_pred_2[:, 1].reshape(-1, 1)


## Subtask 3
print("Subtask 3")
X = imputed_train_feature
y = train_label[SIGN_LABEL].values

n_estimators = 200
n_jobs = -1
clf_3 = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, verbose=0, n_jobs=n_jobs))

clf_3 = subtask3.train(clf_3, X, y)

# Predict
y_pred_3 = clf_3.predict(imputed_test_feature)


## Save prediction result
print("Saving...")
output_df = pd.DataFrame(np.concatenate((y_pred_1, y_pred_2, y_pred_3), axis=1).astype(np.float32), index=test_feature.index,
                         columns=ORDER_LABEL+SEPSIS_LABEL+SIGN_LABEL)
org_test_pid = pd.read_csv(test_filename)['pid'].drop_duplicates().values
sorted_output_df = output_df.loc[org_test_pid, :]

sorted_output_df.to_csv(result_filename, compression='zip')