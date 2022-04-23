from feature_extractor import extract_features
import subtask1
import subtask2
import pickle
import numpy as np

if __name__ == "__main__":
    # Extract features
    print("Extracted features")
    path = "./data/"
    save_to_csv = True
    feat_test = extract_features(path, "test", save2csv=save_to_csv)


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
    pred2 = model2.predict_proba(X2)
    pred2 = np.array(pred2)[:, 1].transpose()  # of shape [n_samples, n_classes]
    print("subtask2 done!")

    # TODO Subtask 3


    # TODO Compact predictions from three subtasks and store it in dataframe


    # TODO Save dataframe to sample.zip




