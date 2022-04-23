import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import os
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def data():
    # read files
    train_data = pd.read_csv('./data/train_feature_extracted.csv').sort_values(by=['pid'])
    indexNames = train_data.columns.values
    #print(indexNames)
    # preprocess data
    datagroup = pd.DataFrame(train_data)#.interpolate()

    dataset = datagroup.groupby(by=['pid'])#.fillna(0)

    #imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
    #imp.fit(datagroup_data)
    #imputed=imp.transform(train_data)
    #store data
    imputeddata = pd.DataFrame(dataset)
    print(datagroup.shape)
    #imputeddata.columns=indexNames
    imputeddata.to_csv("./data/traindata_imputed.csv")

    test_data = pd.read_csv('./data/test_features_extracted.csv').sort_values(by=['pid'])
    test_gro = pd.DataFrame(test_data)
    test_group = test_gro.groupby(by=['pid']).fillna(0)
    imputedtest = pd.DataFrame(test_group)
    #imputedtest.columns = indexNames
    imputedtest.to_csv("./data/testdata_imputed.csv")
    print("Data imputed")

def train2():
    train_data =pd.read_csv("./data/traindata_imputed.csv")
    train_feature = pd.read_csv("./data/train_labels.csv").sort_values(by=["pid"]).fillna(0)
    feature = train_feature["LABEL_Sepsis"]
    testimputed = pd.read_csv("./data/testdata_imputed.csv")#interpolate()
    x = np.array(train_data.values)
    y = np.array(feature.values)
    z = np.array(testimputed.values)
    print(z.shape)
    print(x.shape)
    print(y.shape)

    #X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    #y = np.array([1, 1, 2, 2])
    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #clf.fit(X, y)
    #print(X.shape)
    #print(y.shape)
    #print(clf.predict([[-0.8, -1]]))
    #imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
    #imp.fit(test_data)

    # GPC
    #kernel = 1.0*RBF(1.0)
    #gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    #gpc.fit(train_data,train_feature)
    #result = gpc.predict(test_group)

    # SVC
    #clf = make_pipeline(StandardScaler,SVC(gamma='auto'))
    #clf.fit(x,y)
    #print(clf.fit(x,y))
    #result = clf.predict(z)
    #resultdata = pd.DataFrame(result)
    #resultdata.to_csv("trainresuult.csv")
    seed = 2022
    rf = RandomForestClassifier(random_state=seed, class_weight="balanced_subsample")
    params = {
        "n_estimators": [400, 500, 600],
        "max_depth": [9, 12, 15],
    }
    clf = GridSearchCV(rf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=10, verbose=3, refit=True)
    clf.fit(x, y)
    final_model = clf.best_estimator_
    pred = final_model.predict_proba(x)
    print(pred.shape)
    pred_ = np.array(pred)[ :, 1].transpose()
    task2 = metrics.roc_auc_score(y, pred_)
    print(f"Overall Train roc_auc: {task2}")
    return final_model.predict(z)


if __name__ == "__main__":
    print("Subtask 2")
    data()
    z =train2()
    dbz = pd.DataFrame(z)
    dbz.to_csv('task2.csv')