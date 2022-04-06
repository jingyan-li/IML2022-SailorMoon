import os.path
import numpy as np
import pandas as pd


def trend(y):
    '''
    Fit 24h data in linear model and calculate slope
    :param y:
    :return:
    '''
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m,c = np.linalg.lstsq(X, y, rcond=None)[0]
    return m


def intercept(y):
    '''
    Fit 24h data in linear model and calculate intercept
    :param y:
    :return:
    '''
    X = np.arange(len(y))
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    X = np.vstack([X, np.ones(len(X))]).T
    m,c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c


def extract_features(path="./data/", split="train", save2csv=True):
    """
    Extract features and save as csv (if required)
    :return: features (pd.Dataframe)
    """
    data = pd.read_csv(os.path.join(path, f"{split}_features.csv"))
    # age
    ages = data[['pid', 'Age']].groupby('pid').min()
    # the other tests/signs
    data_signals = data.drop(axis='columns', labels=['Time', 'Age'], inplace=False)
    feats = data_signals.groupby(by="pid").agg(['min', 'max', 'std', 'mean', 'median', 'count', trend, intercept])
    feats.columns = feats.columns.to_flat_index()
    # Combine features & age
    feats_withage = feats.join(ages, on='pid')
    # Save to csv if needed
    if save2csv:
        feats_withage.to_csv(os.path.join(path, f"{split}_feature_extracted.csv"))
    return feats_withage


if __name__ == "__main__":
    path = "./data/"
    split = "test"
    extract_features(path, split, save2csv=True)



