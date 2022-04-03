from feature_extractor import extract_features

if __name__ == "__main__":
    # Extract features
    path = "./data/"
    save_to_csv = True
    feat_train = extract_features(path, "train", save2csv=save_to_csv)
    feat_test = extract_features(path, "test", save2csv=save_to_csv)
    # Train



    # Predict