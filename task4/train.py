"""
Scripts for training
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import argparse
import torch
import yaml
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import os

from src.PretrainDataset import MolecularData
from src.PretrainModel import PretrainEmbeddingNet


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--pretrain_config", help="path to pretrain config", default='configs/pretrain.yaml')
    a.add_argument("--train_config", help="path to train config", default='configs/train.yaml')
    args = a.parse_args()
    return args


if __name__ == "__main__":
    #################### Read Configs ##########################
    args = config()
    with open(args.pretrain_config, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    with open(args.train_config, "r") as yamlfile:
        train_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Output of Training
    out_root = Path(train_config["output_root"]) / train_config["pretrain_checkpoint"].split("/")[-1]
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    #################### Load Pretrain Model ##########################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Initialize pretrain model
    pretrain_model = PretrainEmbeddingNet(**config['pretrain_model_cfg'])

    # To gpu
    pretrain_model = pretrain_model.to(device)

    # Load Checkpoints
    checkpoint_path = Path(train_config['pretrain_checkpoint']) / 'checkpoints' / 'best'
    pretrain_model.load_state_dict(torch.load(checkpoint_path)['model'])
    pretrain_model.eval()
    print("Pretrain model loaded!")

    #################### Extract embeddings for train / test ##########################
    # Set up datasets
    data = MolecularData(load_sets=['train', 'test'], train_val_split=False, **config["data_cfg"])
    # Run model to get embedding for train
    pred_embedding = []
    pred_y_target = []
    with torch.no_grad():
        for step, sample_dict in enumerate(data.dataloaders["train"]):
            feature, target, indices = sample_dict["feature"], sample_dict["label"], sample_dict["index"]
            embedding = pretrain_model.get_embedding(feature.to(device))
            pred_embedding.append(embedding)
            pred_y_target.append(target)
    pred_embedding = torch.cat(pred_embedding, 0).cpu().numpy()
    pred_y_target = torch.cat(pred_y_target, 0).numpy()
    print("Feature extracted: train ready!")

    # Run model to get embedding for test
    test_embedding = []
    test_y_idx = []
    with torch.no_grad():
        for step, sample_dict in enumerate(data.dataloaders["test"]):
            feature, indices = sample_dict["feature"], sample_dict["index"]
            embedding = pretrain_model.get_embedding(feature.to(device))
            test_embedding.append(embedding)
            test_y_idx.append(indices)
    test_embedding = torch.cat(test_embedding, 0).cpu().numpy()
    test_y_idx = torch.cat(test_y_idx, 0).numpy()
    print("Feature extracted: test ready!")

    #################### Training ##########################
    # Fit the model
    ## TODO: Add Cross Validation /Hyperparameter tuning/ Try other models
    train_model = RandomForestRegressor(n_estimators=100, random_state=config['train_cfg']['random_seed'])
    # Multiply by -1 since sklearn calculates *negative* RMSE
    scores = -1 * cross_val_score(train_model, pred_embedding, pred_y_target.ravel(),
                                  cv=5,
                                  scoring='neg_root_mean_squared_error')
    # Average over cv results
    scores_avg = np.mean(scores)
    print(f"TRAIN DATA - CV MSE {scores_avg}; scores {scores}")

    #################### Inference on Test data ##########################
    # Predict by model
    train_model.fit(X=pred_embedding, y=pred_y_target.ravel())
    test_y_target = train_model.predict(test_embedding)
    # Combine target and index
    test_result_df = pd.DataFrame({"Id": test_y_idx, "y": test_y_target}).sort_values(by="Id")
    # Save to csv
    test_result_df.to_csv(out_root/"submit.csv", index=False)
    print("Test prediction saved!")
