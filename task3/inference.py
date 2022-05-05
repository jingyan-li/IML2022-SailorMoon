"""
Main script for training
"""

import argparse

import torch
import os
import numpy as np
import wandb
import yaml
from tqdm import tqdm

from Fooddataset import FoodData
from SiameseNetwork import EmbeddingNet, EmbeddingNetL2, TripletModel


def config():
    a = argparse.ArgumentParser()
    a.add_argument("--train_config", help="path to train config", default='configs/train.yaml')
    a.add_argument("--pred_config", help="path to inference config", default="configs/inference.yaml")
    args = a.parse_args()
    return args


if __name__ == "__main__":
    args = config()

    with open(args.train_config) as fh:
        train_config = yaml.safe_load(fh)

    with open(args.pred_config) as fh:
        pred_config = yaml.safe_load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Initialize model
    img_size = train_config['data_cfg']['image_resize']
    embedding_model = EmbeddingNet(img_size, **train_config['model_cfg']['embedding_model']).to(device)
    triplet_model = TripletModel(embedding_model, **train_config['model_cfg']['triplet_model']).to(device)

    checkpoint_path = os.path.join(pred_config['inference_cfg']['resume_checkpoint'], 'checkpoints/best')
    triplet_model.load_state_dict(torch.load(checkpoint_path)['model'])
    triplet_model.eval()

    # create dataset
    data = FoodData(load_sets=['test'], **pred_config['data_cfg'])
    dataloader = data.dataloaders['test']

    # run model
    results = []
    for step, batch in enumerate(tqdm(dataloader)):
        anchor_img, positive_img, negative_img = (_.to(device) for _ in batch)
        B, C, W, H = anchor_img.shape
        # print(f"Batch size {B}")
        pred_y, loss = triplet_model.predict(anchor_img, positive_img, negative_img)
        results.append(pred_y.to('cpu').numpy())
        # print(f"Batch {step} - Loss {loss} - pred_y {results[-1]}")

    results = np.asarray(results).reshape(-1).tolist()
    # Save results
    with open("predict.csv", "w") as file:
        file.writelines("\n".join([str(int(_)) for _ in results]))
