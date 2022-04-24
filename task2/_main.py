import pandas as pd

import subtask3
from feature_extractor import extract_features
import subtask1
import subtask2
import pickle
import numpy as np
import Labels

if __name__ == "__main__":
    # ########## Preprocess ############
    # Extract features
    print("Extracted features")
    path = "./data/"
    save_to_csv = True
    feat_test = extract_features(path, "test", save2csv=save_to_csv)
    pids_test = feat_test['pid'].values

    # ########## Make Prediction ############
    # Subtask1
    print("subtask 1")
    # Get test features
    X1, _ = subtask1.get_features(path=path, split='test')
    # Use best estimator & params
    model1 = pickle.load(open('./log/subtask1_best.p', "rb"))
    # Predict
    pred1 = model1.predict_proba(X1)
    pred1 = np.array(pred1)[:, :, 1].transpose() # of shape [n_samples, n_classes]
    print("subtask1 done!")
    del X1

    # Subtask 2
    print("subtask 2")
    # Use best estimator & params
    X2, _ = subtask2.get_features(path=path, split='test')
    model2 = pickle.load(open('./log/subtask2_best.p', "rb"))
    # Predict
    pred2 = model2.predict_proba(X2)[:, 1:] # of shape [n_samples, n_classes]
    print("subtask2 done!")
    del X2

    # Subtask 3
    X3, _ = subtask3.get_features(path=path, split='test')
    model3 = pickle.load(open('./log/subtask3_best.p', "rb"))
    # Predict
    pred3 = model3.predict(X3) # of shape [n_samples, n_classes]
    print("subtask3 done!")
    del X3

    # ########## Compact Prediction & Save ############
    print(f"Saving result to {path}/submit.csv...")
    # Compact predictions from three subtasks and store it in dataframe
    pred = np.hstack([pids_test[:, np.newaxis], pred1, pred2, pred3]).astype(np.float32)
    print(f"Prediction array shape : {pred.shape}")
    pred_df = pd.DataFrame(data=pred, columns=["pid"] + Labels.VITALS + Labels.TESTS + Labels.SEPSIS)
    pred_df.set_index("pid", drop=True, inplace=True)
    # Save dataframe to sample.zip
    pred_df.to_csv(path+"submit.csv.zip")



